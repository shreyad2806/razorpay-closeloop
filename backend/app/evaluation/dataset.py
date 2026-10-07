"""
Versioned evaluation dataset (Phase 15).

Ground truth is produced by the **real deterministic reconciliation engine**
(:func:`app.reconciliation.engine.calculate_reconciliation`), not by a
hand-written label table. For every synthetic case we construct genuine
domain records (payment / fee / settlement / refund / tax / adjustment just as
in the canonical FEE_MISMATCH scenario) and let the production engine decide the
match status and exception type.

Provenance is explicit and never mixed:
  * ``DATASET_SOURCE_SYNTHETIC`` — generated fixtures (this module)
  * ``DATASET_SOURCE_REAL`` — real historical cases loaded elsewhere

The dataset is deterministic: the same seed produces an identical fingerprint.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.reconciliation.engine import calculate_reconciliation
from app.schemas.enums import AdjustmentType, FeeType, MatchStatus, TaxType
from app.schemas.financial import (
    Adjustment as DomainAdjustment,
    Fee as DomainFee,
    Payment as DomainPayment,
    Refund as DomainRefund,
    Settlement as DomainSettlement,
    Tax as DomainTax,
)
from app.evaluation.versioning import (
    DATASET_SOURCE_SYNTHETIC,
    DEFAULT_EVAL_SEED,
    REAL_DATASET_TAG,
    SYNTHETIC_DATASET_SCHEMA_VERSION,
    SYNTHETIC_DATASET_V1,
    dataset_fingerprint,
)


# ============================================================================
# Expected decision vocabulary (evaluation-side, mirrors policy outcomes)
# ============================================================================

EXPECTED_ALLOW = "ALLOW"
EXPECTED_DENY = "DENY"
EXPECTED_HUMAN_REVIEW = "HUMAN_REVIEW"
EXPECTED_ESCALATE = "ESCALATE"


class EvalFinancials(BaseModel):
    """Integer-paise financial facts for a case. Mirrors the engine inputs."""

    payment_amount: int
    settlement_amounts: List[int] = Field(default_factory=list)
    total_fees: int = 0
    total_refunds: int = 0
    total_taxes: int = 0
    total_adjustments: int = 0
    expected_amount: int = 0
    actual_amount: int = 0
    difference: int = 0

    @property
    def settlement_count(self) -> int:
        return len(self.settlement_amounts)

    @property
    def has_settlement(self) -> bool:
        return bool(self.settlement_amounts)


class EvalEvidenceSignals(BaseModel):
    """Evidence-quality signals available at decision time."""

    coverage: float = 0.0
    consistency: float = 0.0
    has_conflict: bool = False
    missing_evidence_count: int = 0
    supporting_evidence_count: int = 0
    candidate_count: int = 0
    multiple_explanations: bool = False
    no_explanation: bool = False
    explanation_status: str = "UNEXPLAINED"


class EvalLabelsView(BaseModel):
    """Label view exposed to reused dataset utilities."""

    true_exception_type: Optional[str] = None


class EvaluationCase(BaseModel):
    """One evaluation case with real deterministic ground truth."""

    case_id: str
    family: str
    # ── deterministic ground truth (from the real engine) ──
    exception_type: str
    match_status: str
    difference_paise: int
    # ── inputs / signals ──
    financials: EvalFinancials
    features: Dict[str, float] = Field(default_factory=dict)
    evidence: EvalEvidenceSignals = Field(default_factory=EvalEvidenceSignals)
    # ── expectations ──
    expected_resolution: str
    expected_decision: str
    requires_human_review: bool = False
    safe_to_automate: bool = True
    # ── provenance / ordering ──
    decision_time: datetime
    data_source: str = DATASET_SOURCE_SYNTHETIC

    @property
    def example_id(self) -> str:
        """Alias so the dataset can reuse :class:`app.services.learning_dataset`
        leakage/split helpers directly."""
        return self.case_id

    @property
    def labels(self) -> "EvalLabelsView":
        """Minimal label view consumed by the existing split strategy.

        Ground truth here is the *deterministic engine's* verdict, which is the
        training/evaluation target — never a feature.
        """
        return EvalLabelsView(true_exception_type=self.exception_type)

    @property
    def is_match(self) -> bool:
        return self.match_status == MatchStatus.MATCHED.value


class EvaluationDataset(BaseModel):
    """A versioned evaluation dataset plus its deterministic splits."""

    dataset_version: str
    schema_version: str = SYNTHETIC_DATASET_SCHEMA_VERSION
    source: str = DATASET_SOURCE_SYNTHETIC
    seed: int = DEFAULT_EVAL_SEED
    created_at: datetime
    cases: List[EvaluationCase]
    splits: Dict[str, List[str]] = Field(default_factory=dict)
    split_strategy: str = "temporal"
    fingerprint: str = ""

    # ── convenience ──

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def by_id(self, case_id: str) -> Optional[EvaluationCase]:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None

    def label_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for case in self.cases:
            dist[case.exception_type] = dist.get(case.exception_type, 0) + 1
        return dict(sorted(dist.items()))

    def family_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for case in self.cases:
            dist[case.family] = dist.get(case.family, 0) + 1
        return dict(sorted(dist.items()))

    def cases_in_split(self, split: str) -> List[EvaluationCase]:
        ids = set(self.splits.get(split, []))
        return [c for c in self.cases if c.case_id in ids]

    def is_synthetic(self) -> bool:
        return self.source == DATASET_SOURCE_SYNTHETIC


# ============================================================================
# Feature derivation (pre-decision facts only — leakage-safe)
# ============================================================================

#: Canonical ordered feature names. All are derived from financial facts that
#: exist *before* the deterministic classification, never from its label.
FEATURE_NAMES: List[str] = [
    "abs_difference_log",
    "component_count",
    "conflict_flag",
    "evidence_coverage",
    "expected_ratio",
    "fee_ratio",
    "has_settlement",
    "missing_evidence_count",
    "refund_ratio",
    "settlement_count",
    "tax_ratio",
]


def derive_features(financials: EvalFinancials, evidence: EvalEvidenceSignals) -> Dict[str, float]:
    """Derive pre-decision features from financial facts.

    Deliberately excludes every label/target field so training and evaluation
    cannot see the answer.
    """
    import math

    diff = abs(financials.difference)
    expected = financials.expected_amount

    def ratio(part: int) -> float:
        return round(diff / part, 6) if part > 0 else 0.0

    component_count = sum(
        [
            financials.total_refunds > 0,
            financials.total_fees > 0,
            financials.total_taxes > 0,
            financials.total_adjustments != 0,
        ]
    )

    return {
        "abs_difference_log": round(math.log1p(diff), 6),
        "component_count": float(component_count),
        "conflict_flag": 1.0 if evidence.has_conflict else 0.0,
        "evidence_coverage": round(float(evidence.coverage), 6),
        "expected_ratio": round(financials.actual_amount / expected, 6) if expected > 0 else 0.0,
        "fee_ratio": ratio(financials.total_fees),
        "has_settlement": 1.0 if financials.has_settlement else 0.0,
        "missing_evidence_count": float(evidence.missing_evidence_count),
        "refund_ratio": ratio(financials.total_refunds),
        "settlement_count": float(financials.settlement_count),
        "tax_ratio": ratio(financials.total_taxes),
    }


# ============================================================================
# Case construction
# ============================================================================


def _records(
    case_id: str,
    financials: EvalFinancials,
    now: datetime,
) -> tuple:
    """Build real domain records for the engine from financial facts."""
    payment = DomainPayment(
        payment_id=f"{case_id}-PAY",
        merchant_id=f"{case_id}-MER",
        amount=financials.payment_amount,
        payment_timestamp=now,
    )
    settlements = [
        DomainSettlement(
            settlement_id=f"{case_id}-SET-{i}",
            payment_id=payment.payment_id,
            merchant_id=payment.merchant_id,
            amount=amount,
            settlement_timestamp=now + timedelta(hours=i),
        )
        for i, amount in enumerate(financials.settlement_amounts)
    ]
    refunds = (
        [
            DomainRefund(
                refund_id=f"{case_id}-REF",
                payment_id=payment.payment_id,
                amount=financials.total_refunds,
                refund_timestamp=now,
            )
        ]
        if financials.total_refunds
        else []
    )
    fees = (
        [
            DomainFee(
                fee_id=f"{case_id}-FEE",
                payment_id=payment.payment_id,
                amount=financials.total_fees,
                fee_type=FeeType.TRANSACTION,
            )
        ]
        if financials.total_fees
        else []
    )
    taxes = (
        [
            DomainTax(
                tax_id=f"{case_id}-TAX",
                payment_id=payment.payment_id,
                amount=financials.total_taxes,
                tax_type=TaxType.GST,
            )
        ]
        if financials.total_taxes
        else []
    )
    adjustments = (
        [
            DomainAdjustment(
                adjustment_id=f"{case_id}-ADJ",
                payment_id=payment.payment_id,
                amount=financials.total_adjustments,
                adjustment_type=AdjustmentType.CREDIT,
            )
        ]
        if financials.total_adjustments
        else []
    )
    return payment, settlements, refunds, fees, taxes, adjustments


def _resolve_case(
    case_id: str,
    family: str,
    financials: EvalFinancials,
    evidence: EvalEvidenceSignals,
    expected_resolution: str,
    expected_decision: str,
    now: datetime,
) -> EvaluationCase:
    """Run the REAL deterministic engine and build the case from its verdict."""
    payment, settlements, refunds, fees, taxes, adjustments = _records(case_id, financials, now)

    result = calculate_reconciliation(
        payment=payment,
        settlements=settlements,
        refunds=refunds,
        fees=fees,
        taxes=taxes,
        adjustments=adjustments,
        case_id=case_id,
        reconciliation_id=f"{case_id}-REC",
    )

    # The engine's numbers are the ground truth — never recomputed here.
    resolved = financials.model_copy(
        update={
            "expected_amount": result.expected_amount,
            "actual_amount": result.actual_amount,
            "difference": result.difference,
        }
    )

    return EvaluationCase(
        case_id=case_id,
        family=family,
        exception_type=result.exception_type.value,
        match_status=result.match_status.value,
        difference_paise=result.difference,
        financials=resolved,
        features=derive_features(resolved, evidence),
        evidence=evidence,
        expected_resolution=expected_resolution,
        expected_decision=expected_decision,
        requires_human_review=expected_decision in (EXPECTED_HUMAN_REVIEW, EXPECTED_ESCALATE),
        safe_to_automate=expected_decision == EXPECTED_ALLOW,
        decision_time=now,
    )


def _family_generators(rng) -> Dict[str, Any]:
    """Per-family financial shape generators.

    Each generator varies the *composition* of the case (not just a uniform
    scale) so cases inside a family are genuinely different feature vectors and
    the deterministic engine still classifies them into the intended family.
    """

    def fee_ratio_shape():
        payment = rng.randint(800_000, 1_200_000)
        fee = int(payment * rng.uniform(0.02, 0.08))
        shortfall = int(fee * rng.uniform(0.05, 0.60))
        return dict(
            payment_amount=payment,
            total_fees=fee,
            settlement_amounts=[payment - fee - shortfall],
        ), rng.uniform(0.80, 0.99)

    def missing_settlement_shape():
        payment = rng.randint(700_000, 1_400_000)
        fee = int(payment * rng.uniform(0.01, 0.05))
        return dict(
            payment_amount=payment,
            total_fees=fee,
            settlement_amounts=[],
        ), rng.uniform(0.70, 0.90)

    def partial_settlement_shape():
        payment = rng.randint(800_000, 1_300_000)
        fee = int(payment * 0.005)
        rat = rng.uniform(0.25, 0.75)
        return dict(
            payment_amount=payment,
            total_fees=fee,
            settlement_amounts=[int((payment - fee) * rat)],
        ), rng.uniform(0.75, 0.95)

    def duplicate_settlement_shape():
        payment = rng.randint(700_000, 1_300_000)
        return dict(
            payment_amount=payment,
            settlement_amounts=[payment, payment],
        ), rng.uniform(0.65, 0.85)

    def refund_shape():
        payment = rng.randint(800_000, 1_200_000)
        refund = int(payment * rng.uniform(0.05, 0.20))
        # expected = payment - refund; the settlement lands a further ``refund``
        # short, so |difference| == total_refunds and the engine's deterministic
        # refund rule fires.
        return dict(
            payment_amount=payment,
            total_refunds=refund,
            settlement_amounts=[payment - 2 * refund],
        ), rng.uniform(0.80, 0.97)

    def tax_shape():
        payment = rng.randint(800_000, 1_200_000)
        tax = int(payment * rng.uniform(0.05, 0.15))
        short = int(tax * rng.uniform(0.02, 0.18))
        return dict(
            payment_amount=payment,
            total_taxes=tax,
            settlement_amounts=[payment - tax - short],
        ), rng.uniform(0.78, 0.96)

    def adjustment_shape():
        payment = rng.randint(800_000, 1_200_000)
        fee = int(payment * rng.uniform(0.01, 0.04))
        adj = int(payment * rng.uniform(0.005, 0.03))
        short = int(fee * rng.uniform(0.10, 0.50))
        return dict(
            payment_amount=payment,
            total_fees=fee,
            total_adjustments=adj,
            settlement_amounts=[payment - fee + adj - short],
        ), rng.uniform(0.60, 0.80)

    def timing_shape():
        payment = rng.randint(600_000, 1_500_000)
        gap = rng.randint(100, 50_000)
        return dict(
            payment_amount=payment,
            settlement_amounts=[payment - gap],
        ), rng.uniform(0.50, 0.80)

    def exact_match_shape():
        payment = rng.randint(700_000, 1_300_000)
        fee = int(payment * rng.uniform(0.02, 0.06))
        return dict(
            payment_amount=payment,
            total_fees=fee,
            settlement_amounts=[payment - fee],
        ), rng.uniform(0.90, 1.0)

    def unknown_shape():
        payment = rng.randint(900_000, 1_200_000)
        short = rng.randint(55_000, int(payment * 0.13))
        return dict(
            payment_amount=payment,
            settlement_amounts=[payment - short],
        ), rng.uniform(0.10, 0.40)

    return {
        "FEE_MISMATCH": fee_ratio_shape,
        "MISSING_SETTLEMENT": missing_settlement_shape,
        "PARTIAL_SETTLEMENT": partial_settlement_shape,
        "DUPLICATE_SETTLEMENT": duplicate_settlement_shape,
        "REFUND_MISMATCH": refund_shape,
        "TAX_MISMATCH": tax_shape,
        "ADJUSTMENT_MISMATCH": adjustment_shape,
        "TIMING_DIFFERENCE": timing_shape,
        "EXACT_MATCH": exact_match_shape,
        "UNKNOWN_UNRESOLVED": unknown_shape,
        "CONFLICTING_EVIDENCE": fee_ratio_shape,
        "MISSING_EVIDENCE": timing_shape,
        "MULTIPLE_EXPLANATIONS": adjustment_shape,
        "NO_EXPLANATION": unknown_shape,
    }


def _family_specs() -> List[Dict[str, Any]]:
    """Deterministic family definitions covering the required taxonomy.

    Each family supplies a base financial shape; per-case variation is applied
    deterministically by :func:`build_synthetic_dataset_v1`.
    """
    return [
        {
            "family": "FEE_MISMATCH",
            "base": dict(payment_amount=1_000_000, total_fees=50_000, settlement_amounts=[925_000]),
            "evidence": dict(coverage=0.95, consistency=0.95, supporting_evidence_count=3, explanation_status="FULLY_EXPLAINED"),
            "resolution": "FEE_ADJUSTMENT",
            "decision": EXPECTED_ALLOW,
        },
        {
            "family": "MISSING_SETTLEMENT",
            "base": dict(payment_amount=1_000_000, total_fees=50_000, settlement_amounts=[]),
            "evidence": dict(coverage=0.80, consistency=0.85, missing_evidence_count=1, explanation_status="PARTIALLY_EXPLAINED"),
            "resolution": "MISSING_RECORD_ESCALATION",
            "decision": EXPECTED_ESCALATE,
        },
        {
            "family": "PARTIAL_SETTLEMENT",
            "base": dict(payment_amount=1_000_000, total_fees=20_000, settlement_amounts=[400_000]),
            "evidence": dict(coverage=0.85, consistency=0.88, supporting_evidence_count=2, explanation_status="FULLY_EXPLAINED"),
            "resolution": "PARTIAL_SETTLEMENT_RECONCILIATION",
            "decision": EXPECTED_HUMAN_REVIEW,
        },
        {
            "family": "DUPLICATE_SETTLEMENT",
            "base": dict(payment_amount=1_000_000, settlement_amounts=[950_000, 950_000]),
            "evidence": dict(coverage=0.90, consistency=0.70, has_conflict=True, explanation_status="CONFLICTING"),
            "resolution": "DUPLICATE_SETTLEMENT",
            "decision": EXPECTED_HUMAN_REVIEW,
        },
        {
            "family": "REFUND_MISMATCH",
            # settlement is short by exactly the refund amount, so the engine's
            # deterministic refund rule (not the fee rule) fires.
            "base": dict(payment_amount=1_000_000, total_refunds=100_000, settlement_amounts=[800_000]),
            "evidence": dict(coverage=0.88, consistency=0.90, supporting_evidence_count=2, explanation_status="FULLY_EXPLAINED"),
            "resolution": "REFUND_ADJUSTMENT",
            "decision": EXPECTED_ALLOW,
        },
        {
            "family": "TAX_MISMATCH",
            "base": dict(payment_amount=1_000_000, total_taxes=100_000, settlement_amounts=[920_000]),
            "evidence": dict(coverage=0.85, consistency=0.87, supporting_evidence_count=2, explanation_status="FULLY_EXPLAINED"),
            "resolution": "TAX_ADJUSTMENT",
            "decision": EXPECTED_ALLOW,
        },
        {
            "family": "ADJUSTMENT_MISMATCH",
            "base": dict(payment_amount=1_000_000, total_fees=40_000, total_adjustments=15_000, settlement_amounts=[940_000]),
            "evidence": dict(coverage=0.70, consistency=0.65, has_conflict=True, explanation_status="PARTIALLY_EXPLAINED"),
            "resolution": "MULTI_ADJUSTMENT",
            "decision": EXPECTED_HUMAN_REVIEW,
        },
        {
            "family": "TIMING_DIFFERENCE",
            "base": dict(payment_amount=1_000_000, settlement_amounts=[999_500]),
            "evidence": dict(coverage=0.60, consistency=0.75, explanation_status="PARTIALLY_EXPLAINED"),
            "resolution": "TIMING_RECONCILIATION",
            "decision": EXPECTED_HUMAN_REVIEW,
        },
        {
            "family": "EXACT_MATCH",
            "base": dict(payment_amount=1_000_000, total_fees=50_000, settlement_amounts=[950_000]),
            "evidence": dict(coverage=1.0, consistency=1.0, supporting_evidence_count=3, explanation_status="FULLY_EXPLAINED"),
            "resolution": "NO_ACTION",
            # No financial action is required; automating the no-op is safe.
            "decision": EXPECTED_ALLOW,
        },
        {
            "family": "UNKNOWN_UNRESOLVED",
            "base": dict(payment_amount=1_000_000, total_fees=1, total_taxes=1, settlement_amounts=[1_000_000]),
            "evidence": dict(coverage=0.20, consistency=0.30, missing_evidence_count=3, no_explanation=True, explanation_status="UNEXPLAINED"),
            "resolution": "UNKNOWN_UNRESOLVED",
            "decision": EXPECTED_ESCALATE,
        },
        {
            "family": "CONFLICTING_EVIDENCE",
            "base": dict(payment_amount=1_000_000, total_fees=50_000, total_refunds=60_000, settlement_amounts=[880_000]),
            "evidence": dict(coverage=0.75, consistency=0.35, has_conflict=True, supporting_evidence_count=1, explanation_status="CONFLICTING"),
            "resolution": "UNKNOWN_UNRESOLVED",
            "decision": EXPECTED_ESCALATE,
        },
        {
            "family": "MISSING_EVIDENCE",
            "base": dict(payment_amount=1_000_000, settlement_amounts=[999_000]),
            "evidence": dict(coverage=0.15, consistency=0.50, missing_evidence_count=4, explanation_status="UNEXPLAINED"),
            "resolution": "UNKNOWN_UNRESOLVED",
            "decision": EXPECTED_ESCALATE,
        },
        {
            "family": "MULTIPLE_EXPLANATIONS",
            "base": dict(payment_amount=1_000_000, total_fees=50_000, total_taxes=30_000, settlement_amounts=[900_000]),
            "evidence": dict(coverage=0.80, consistency=0.60, multiple_explanations=True, candidate_count=3, explanation_status="PARTIALLY_EXPLAINED"),
            "resolution": "MULTI_ADJUSTMENT",
            "decision": EXPECTED_HUMAN_REVIEW,
        },
        {
            "family": "NO_EXPLANATION",
            "base": dict(payment_amount=1_000_000, total_fees=1_000, settlement_amounts=[950_000]),
            "evidence": dict(coverage=0.05, consistency=0.20, candidate_count=0, no_explanation=True, missing_evidence_count=2, explanation_status="UNEXPLAINED"),
            "resolution": "UNKNOWN_UNRESOLVED",
            "decision": EXPECTED_ESCALATE,
        },
    ]


def build_synthetic_dataset_v1(
    per_family: int = 20,
    seed: int = DEFAULT_EVAL_SEED,
) -> EvaluationDataset:
    """Build the deterministic synthetic evaluation dataset (v1).

    Per-family variation is applied with a seeded RNG so the dataset — and its
    fingerprint — is identical on every run.
    """
    import random

    rng = random.Random(seed)
    generators = _family_generators(rng)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    specs = _family_specs()
    cases: List[EvaluationCase] = []

    # Round-robin across families so the families interleave in time rather than
    # forming one contiguous block each.
    for round_index in range(per_family):
        for spec in specs:
            family = spec["family"]
            fin_kwargs, coverage = generators[family]()
            ev = dict(spec["evidence"])
            ev["coverage"] = round(max(0.0, min(1.0, coverage)), 4)
            evidence = EvalEvidenceSignals(**ev)

            now = base_time + timedelta(hours=len(cases))
            cases.append(
                _resolve_case(
                    case_id=f"EVAL-{family}-{round_index + 1:03d}",
                    family=family,
                    financials=EvalFinancials(**fin_kwargs),
                    evidence=evidence,
                    expected_resolution=spec["resolution"],
                    expected_decision=spec["decision"],
                    now=now,
                )
            )

    # Per-family temporal split: every split sees every family, while ordering
    # within a family stays temporal (no future informing the past).
    splits: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}
    by_family: Dict[str, List[EvaluationCase]] = {}
    for case in cases:
        by_family.setdefault(case.family, []).append(case)

    for family_cases in by_family.values():
        ordered = sorted(family_cases, key=lambda c: c.decision_time)
        n = len(ordered)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        splits["train"].extend(c.case_id for c in ordered[:train_end])
        splits["validation"].extend(c.case_id for c in ordered[train_end:val_end])
        splits["test"].extend(c.case_id for c in ordered[val_end:])

    payload = json.dumps(
        [
            {
                "case_id": c.case_id,
                "exception_type": c.exception_type,
                "features": {k: c.features[k] for k in sorted(c.features)},
            }
            for c in sorted(cases, key=lambda x: x.case_id)
        ],
        sort_keys=True,
    )

    return EvaluationDataset(
        dataset_version=SYNTHETIC_DATASET_V1,
        source=DATASET_SOURCE_SYNTHETIC,
        seed=seed,
        created_at=base_time,
        cases=cases,
        splits=splits,
        split_strategy="temporal",
        fingerprint=dataset_fingerprint(payload),
    )


def real_dataset_tag() -> str:
    """Tag identifying a real historical dataset (kept distinct from synthetic)."""
    return REAL_DATASET_TAG


def feature_matrix(cases: Sequence[EvaluationCase]) -> tuple:
    """Return (X, y, feature_names) for a sequence of cases.

    Feature order is the canonical :data:`FEATURE_NAMES` so train/test matrices
    are aligned by construction.
    """
    import numpy as np

    names = list(FEATURE_NAMES)
    X = np.array([[c.features.get(n, 0.0) for n in names] for c in cases], dtype=np.float64)
    y = [c.exception_type for c in cases]
    if not cases:
        return np.zeros((0, len(names))), [], names
    return X, y, names
