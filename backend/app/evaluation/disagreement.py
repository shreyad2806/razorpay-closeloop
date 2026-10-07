"""
ML vs deterministic truth (Phase 15).

Deterministic financial truth is authoritative. This module measures how often
the ML classifier agrees with it and *categorises* the disagreements. It never
resolves a disagreement — the deterministic engine always wins.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import BaseModel, Field

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.metrics import round4, safe_rate
from app.evaluation.ml_eval import predictions_for_cases
from app.evaluation.versioning import DEFAULT_EVAL_SEED, EVALUATOR_VERSION

#: Families that the deterministic engine resolves through the fee channel.
FEE_CHANNEL = {"FEE_MISMATCH", "FEE_DIFFERENCE"}
REFUND_CHANNEL = {"REFUND_ADJUSTMENT"}
TAX_CHANNEL = {"TAX_ADJUSTMENT"}
STRUCTURAL_CHANNEL = {"MISSING_RECORD", "DUPLICATE"}
TIMING_CHANNEL = {"TIMING_DIFFERENCE", "PARTIAL_SETTLEMENT"}
OTHER_CHANNEL = {"EXACT_MATCH", "UNKNOWN", "COMPLEX_MULTI_ADJUSTMENT"}


def _channel(label: str) -> str:
    for name, group in (
        ("FEE", FEE_CHANNEL),
        ("REFUND", REFUND_CHANNEL),
        ("TAX", TAX_CHANNEL),
        ("STRUCTURAL", STRUCTURAL_CHANNEL),
        ("TIMING", TIMING_CHANNEL),
    ):
        if label in group:
            return name
    return "OTHER"


class DisagreementRecord(BaseModel):
    case_id: str
    deterministic_type: str
    ml_predicted_type: str
    confidence: float
    deterministic_channel: str
    ml_channel: str
    category: str


class DisagreementReport(BaseModel):
    evaluator_version: str = EVALUATOR_VERSION
    dataset_version: str = ""
    seed: int = DEFAULT_EVAL_SEED
    cases_compared: int = 0
    agreements: int = 0
    disagreements: int = 0
    agreement_rate: float = 0.0
    cross_channel_disagreements: int = 0
    categories: Dict[str, int] = Field(default_factory=dict)
    records: List[DisagreementRecord] = Field(default_factory=list)
    deterministic_authoritative: bool = True

    def summary(self) -> Dict[str, object]:
        return {
            "cases_compared": self.cases_compared,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "agreement_rate": self.agreement_rate,
            "cross_channel_disagreements": self.cross_channel_disagreements,
            "categories": self.categories,
            "deterministic_authoritative": self.deterministic_authoritative,
            "examples": [r.model_dump() for r in self.records[:20]],
        }


def measure_disagreement(
    dataset: EvaluationDataset,
    cases: Sequence[EvaluationCase] | None = None,
    seed: int = DEFAULT_EVAL_SEED,
) -> DisagreementReport:
    """Compare ML predictions against the deterministic engine's verdict."""
    cases = list(cases if cases is not None else dataset.cases_in_split("test"))
    if not cases:
        return DisagreementReport(dataset_version=dataset.dataset_version, seed=seed)

    predictions = predictions_for_cases(dataset, cases, seed=seed)

    agreements = 0
    categories: Dict[str, int] = {}
    records: List[DisagreementRecord] = []
    cross_channel = 0

    for case in cases:
        predicted, confidence = predictions.get(case.case_id, ("UNKNOWN", 0.0))
        if predicted == case.exception_type:
            agreements += 1
            continue

        det_channel = _channel(case.exception_type)
        ml_channel = _channel(predicted)
        category = f"{det_channel}->{ml_channel}" if det_channel != ml_channel else f"WITHIN_{det_channel}"
        if det_channel != ml_channel:
            cross_channel += 1
        categories[category] = categories.get(category, 0) + 1
        records.append(
            DisagreementRecord(
                case_id=case.case_id,
                deterministic_type=case.exception_type,
                ml_predicted_type=predicted,
                confidence=confidence,
                deterministic_channel=det_channel,
                ml_channel=ml_channel,
                category=category,
            )
        )

    records.sort(key=lambda r: r.case_id)

    return DisagreementReport(
        dataset_version=dataset.dataset_version,
        seed=seed,
        cases_compared=len(cases),
        agreements=agreements,
        disagreements=len(records),
        agreement_rate=round4(safe_rate(agreements, len(cases))),
        cross_channel_disagreements=cross_channel,
        categories=dict(sorted(categories.items())),
        records=records,
        deterministic_authoritative=True,
    )
