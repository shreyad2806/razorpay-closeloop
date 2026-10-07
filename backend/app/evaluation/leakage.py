"""
Data-leakage audit (Phase 15).

A high accuracy score computed on a leaked dataset is worthless, so the audit
is a first-class deliverable rather than an afterthought.

Reuses the existing detection primitives in
:mod:`app.services.learning_dataset` (``LeakageDetector``) and adds the checks
that matter for this evaluation layer:

  * no case appears in more than one split
  * no split observes another split's future
  * feature vectors contain no label/target fields
  * no post-resolution information is present in features
  * embedded/derived features carry no target information
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set

from pydantic import BaseModel, Field

from app.services.learning_dataset import LeakageDetector

from app.evaluation.dataset import EvaluationCase, EvaluationDataset, FEATURE_NAMES
from app.evaluation.versioning import EVALUATOR_VERSION


#: Feature names that would leak the target if they ever appeared.
FORBIDDEN_FEATURE_KEYS: Set[str] = {
    "true_exception_type",
    "exception_type",
    "label",
    "labels",
    "target",
    "true_resolution",
    "resolution_type",
    "expected_resolution",
    "match_status",
    "resolvable",
    "risk_category",
    "guardrail_decision",
    "expected_decision",
    "is_correct",
    "predicted_exception_type",
    "verification_passed",
    "human_corrected",
}

#: Fields that only exist *after* a resolution is chosen — using them to predict
#: a pre-resolution outcome is temporally invalid.
POST_RESOLUTION_KEYS: Set[str] = {
    "verification_result",
    "verification_passed",
    "verification_status",
    "execution_status",
    "closed_at",
    "closure_reason",
    "reward_value",
    "human_corrected",
    "provider_result",
    "settled_after",
}

#: Feature names present only in the feature *space*, never targets.
ALLOWED_FEATURE_KEYS: Set[str] = set(FEATURE_NAMES)


class LeakageFinding(BaseModel):
    """A single leakage finding."""

    kind: str
    severity: str
    description: str
    evidence: Dict[str, object] = Field(default_factory=dict)


class LeakageReport(BaseModel):
    """Result of the leakage audit."""

    evaluator_version: str = EVALUATOR_VERSION
    dataset_version: str = ""
    dataset_source: str = ""
    cases_checked: int = 0
    findings: List[LeakageFinding] = Field(default_factory=list)
    leakage_free: bool = True

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    def summary(self) -> Dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "dataset_source": self.dataset_source,
            "cases_checked": self.cases_checked,
            "leakage_free": self.leakage_free,
            "critical_count": self.critical_count,
            "findings": [f.model_dump() for f in self.findings],
        }


def audit_feature_keys(
    cases: Sequence[EvaluationCase],
) -> List[LeakageFinding]:
    """Verify feature vectors carry no label or post-resolution information."""
    findings: List[LeakageFinding] = []
    for case in cases:
        keys = set(case.features)
        leaked = keys & FORBIDDEN_FEATURE_KEYS
        if leaked:
            findings.append(
                LeakageFinding(
                    kind="LABEL_IN_FEATURES",
                    severity="critical",
                    description=f"Case {case.case_id} exposes label fields as features",
                    evidence={"case_id": case.case_id, "fields": sorted(leaked)},
                )
            )
        post = keys & POST_RESOLUTION_KEYS
        if post:
            findings.append(
                LeakageFinding(
                    kind="POST_RESOLUTION_IN_FEATURES",
                    severity="critical",
                    description=f"Case {case.case_id} uses post-resolution information",
                    evidence={"case_id": case.case_id, "fields": sorted(post)},
                )
            )
        unknown = keys - ALLOWED_FEATURE_KEYS
        if unknown:
            findings.append(
                LeakageFinding(
                    kind="UNREGISTERED_FEATURE",
                    severity="warning",
                    description=f"Case {case.case_id} has unregistered features",
                    evidence={"case_id": case.case_id, "fields": sorted(unknown)},
                )
            )
    return findings


def audit_split_disjointness(dataset: EvaluationDataset) -> List[LeakageFinding]:
    """No case may appear in more than one split."""
    train = set(dataset.splits.get("train", []))
    val = set(dataset.splits.get("validation", []))
    test = set(dataset.splits.get("test", []))

    issues = LeakageDetector.check_no_case_overlap(train, val, test)
    return [
        LeakageFinding(
            kind="SPLIT_OVERLAP",
            severity="critical" if issue.severity == "critical" else "warning",
            description=issue.description,
            evidence={"detector_severity": issue.severity},
        )
        for issue in issues
    ]


def audit_temporal_ordering(dataset: EvaluationDataset) -> List[LeakageFinding]:
    """Test cases must not predate training cases."""
    train = set(dataset.splits.get("train", []))
    test = set(dataset.splits.get("test", []))
    issues = LeakageDetector.check_temporal_ordering(list(dataset.cases), train, test)
    return [
        LeakageFinding(
            kind="TEMPORAL_ORDER",
            severity="critical" if issue.severity == "critical" else "warning",
            description=issue.description,
        )
        for issue in issues
    ]


def audit_duplicates(dataset: EvaluationDataset) -> List[LeakageFinding]:
    """Duplicate case ids, and identical feature rows shared across splits.

    A feature row that appears in two splits is the same case seen twice: the
    model can memorise it. Within a single split it is only a warning (two
    genuinely different cases may legitimately share a coarse feature vector).
    """
    findings: List[LeakageFinding] = []

    ids = [case.case_id for case in dataset.cases]
    duplicated_ids = sorted({cid for cid in set(ids) if ids.count(cid) > 1})
    if duplicated_ids:
        findings.append(
            LeakageFinding(
                kind="DUPLICATE_CASE_ID",
                severity="critical",
                description=f"{len(duplicated_ids)} case id(s) appear more than once",
                evidence={"case_ids": duplicated_ids[:20]},
            )
        )

    split_of: Dict[str, str] = {}
    for split, split_ids in dataset.splits.items():
        for case_id in split_ids:
            split_of[case_id] = split

    rows: Dict[tuple, List[str]] = {}
    for case in dataset.cases:
        signature = tuple(sorted(case.features.items()))
        rows.setdefault(signature, []).append(case.case_id)

    for case_ids in rows.values():
        if len(case_ids) < 2:
            continue
        splits = {split_of.get(cid) for cid in case_ids} - {None}
        if len(splits) > 1:
            findings.append(
                LeakageFinding(
                    kind="DUPLICATE_FEATURE_ROW",
                    severity="critical",
                    description=(
                        "Identical feature rows appear in more than one split "
                        f"({sorted(splits)})"
                    ),
                    evidence={"case_ids": case_ids[:20]},
                )
            )
        else:
            findings.append(
                LeakageFinding(
                    kind="REPEATED_FEATURE_ROW",
                    severity="warning",
                    description="Identical feature rows within one split",
                    evidence={"case_ids": case_ids[:20]},
                )
            )

    return findings


def audit_family_isolation(dataset: EvaluationDataset) -> List[LeakageFinding]:
    """A family must not straddle train and test.

    Splitting a deterministic family across train/test would let the model see
    a near-identical sibling of every test case, inflating accuracy.
    """
    findings: List[LeakageFinding] = []
    families: Dict[str, Dict[str, int]] = {}
    for split, ids in dataset.splits.items():
        for case in dataset.cases:
            if case.case_id in set(ids):
                families.setdefault(case.family, {})[split] = (
                    families.setdefault(case.family, {}).get(split, 0) + 1
                )
    for family, counts in families.items():
        if counts.get("train", 0) > 0 and counts.get("test", 0) > 0:
            findings.append(
                LeakageFinding(
                    kind="FAMILY_STRADDLE",
                    severity="warning",
                    description=(
                        f"Family {family} appears in both train and test "
                        f"({counts.get('train')} train / {counts.get('test')} test)"
                    ),
                    evidence={"family": family, "counts": counts},
                )
            )
    return findings


def audit_dataset(dataset: EvaluationDataset) -> LeakageReport:
    """Run every leakage check and summarise the result."""
    findings: List[LeakageFinding] = []
    findings.extend(audit_feature_keys(dataset.cases))
    findings.extend(audit_split_disjointness(dataset))
    findings.extend(audit_temporal_ordering(dataset))
    findings.extend(audit_duplicates(dataset))
    findings.extend(audit_family_isolation(dataset))

    return LeakageReport(
        dataset_version=dataset.dataset_version,
        dataset_source=dataset.source,
        cases_checked=dataset.case_count,
        findings=findings,
        leakage_free=not any(f.severity == "critical" for f in findings),
    )
