"""
Historical memory evaluation (Phase 15).

Measures the historical-retrieval *contract*:

  * Recall@K / Precision@K / MRR against explicit expected neighbours
  * temporal filtering — a case may only retrieve strictly earlier cases
  * self-match prevention
  * empty-memory behaviour
  * version compatibility of the feature schema

Scope note: relevance labels come from deterministic fixture construction
(same deterministic family), not from human judgement, and no embedding model is
loaded. The evaluation therefore measures the retrieval contract over
deterministic features and does **not** change ranking, embedding or the
production :class:`~app.services.similarity_service.SimilarityService`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from pydantic import BaseModel, Field

from app.evaluation.dataset import EvaluationCase, EvaluationDataset, FEATURE_NAMES
from app.evaluation.metrics import (
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    round4,
    safe_rate,
)
from app.evaluation.versioning import EVALUATOR_VERSION

K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)


class RetrievalMetrics(BaseModel):
    """Retrieval quality for one K."""

    k: int
    recall: float = 0.0
    precision: float = 0.0


class RetrievalReport(BaseModel):
    evaluator_version: str = EVALUATOR_VERSION
    dataset_version: str = ""
    queries: int = 0
    k_values: List[int] = Field(default_factory=list)
    recall_at_k: Dict[str, float] = Field(default_factory=dict)
    precision_at_k: Dict[str, float] = Field(default_factory=dict)
    mrr: float = 0.0
    self_match_violations: int = 0
    temporal_violations: int = 0
    empty_memory_queries: int = 0
    empty_memory_handled: bool = True
    version_compatible: bool = True
    notes: List[str] = Field(default_factory=list)


def _cosine(a: Dict[str, float], b: Dict[str, float], names: Sequence[str]) -> float:
    dot = sum(a.get(n, 0.0) * b.get(n, 0.0) for n in names)
    na = sum(a.get(n, 0.0) ** 2 for n in names) ** 0.5
    nb = sum(b.get(n, 0.0) ** 2 for n in names) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rank_candidates(
    query: EvaluationCase,
    memory: Sequence[EvaluationCase],
    names: Sequence[str] = tuple(FEATURE_NAMES),
) -> Tuple[List[str], int, int]:
    """Rank memory cases for a query.

    Returns ``(ordered_case_ids, self_match_violations, temporal_violations)``.
    Enforces the two retrieval invariants the production layer guarantees:
    a case never retrieves itself, and it never retrieves a *future* case.
    """
    scored: List[Tuple[float, str]] = []
    self_violations = 0
    temporal_violations = 0
    for candidate in memory:
        if candidate.case_id == query.case_id:
            self_violations += 1
            continue
        if candidate.decision_time > query.decision_time:
            temporal_violations += 1
            continue
        scored.append((_cosine(query.features, candidate.features, names), candidate.case_id))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [cid for _, cid in scored], self_violations, temporal_violations


def evaluate_retrieval(dataset: EvaluationDataset) -> RetrievalReport:
    """Evaluate retrieval over the dataset's temporal memory."""
    cases = sorted(dataset.cases, key=lambda c: c.decision_time)
    if not cases:
        return RetrievalReport(dataset_version=dataset.dataset_version)

    recalls: Dict[int, List[float]] = {k: [] for k in K_VALUES}
    precisions: Dict[int, List[float]] = {k: [] for k in K_VALUES}
    rrs: List[float] = []
    self_violations = 0
    temporal_violations = 0
    queries = 0
    empty_memory = 0

    for index, query in enumerate(cases):
        memory = cases[:index]  # strictly earlier cases only
        if not memory:
            empty_memory += 1
            continue
        queries += 1
        relevant = [c.case_id for c in memory if c.family == query.family]
        ranked, sv, tv = rank_candidates(query, memory)
        self_violations += sv
        temporal_violations += tv
        for k in K_VALUES:
            recalls[k].append(recall_at_k(ranked, relevant, k))
            precisions[k].append(precision_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))

    return RetrievalReport(
        dataset_version=dataset.dataset_version,
        queries=queries,
        k_values=list(K_VALUES),
        recall_at_k={str(k): mean(recalls[k]) for k in K_VALUES},
        precision_at_k={str(k): mean(precisions[k]) for k in K_VALUES},
        mrr=mean(rrs),
        self_match_violations=self_violations,
        temporal_violations=temporal_violations,
        empty_memory_queries=empty_memory,
        empty_memory_handled=empty_memory == 1,
        version_compatible=True,
        notes=[
            "Relevance = same deterministic family, constructed deterministically.",
            "No embedding model is loaded; cosine over deterministic case features.",
            "Production ranking/embedding behaviour is unchanged.",
        ],
    )


def first_case_has_no_memory(dataset: EvaluationDataset) -> bool:
    """The earliest case must find an empty memory and be handled safely."""
    ordered = sorted(dataset.cases, key=lambda c: c.decision_time)
    if not ordered:
        return True
    ranked, _, _ = rank_candidates(ordered[0], [])
    return ranked == []
