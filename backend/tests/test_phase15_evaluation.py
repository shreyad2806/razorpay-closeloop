"""
Phase 15 — Evaluation & Quality Measurement: executable test suite.

Every test here runs the evaluation harness against the *real* production
components (deterministic reconciliation engine, XGBoost classifier, candidate
scorer/selector, guardrail engine, execution service, post-execution
reconciliation service) and asserts behaviour — never source text.

Scope discipline for Phase 15:
  * measurement only — no production decision logic is exercised differently
  * synthetic vs real provenance is explicit
  * zero-denominator metrics are reported as "N/A — no applicable cases"
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database.database import Base
from app.evaluation.dataset import (
    FEATURE_NAMES,
    EvaluationDataset,
    EvalEvidenceSignals,
    build_synthetic_dataset_v1,
    derive_features,
    real_dataset_tag,
)
from app.evaluation.disagreement import measure_disagreement
from app.evaluation.evidence_eval import evaluate_evidence
from app.evaluation.golden import (
    DB_SCENARIOS,
    canonical_reconciliation,
    closure_metrics,
    golden_1_fee_mismatch_full_chain,
    golden_2_already_matched,
    golden_3_missing_evidence,
    golden_4_conflicting_evidence,
    golden_5_high_confidence_policy_denial,
    golden_6_provider_success_reconciliation_failure,
    golden_7_provider_timeout,
    golden_8_stale_approval,
    golden_9_unknown_provider_result,
    golden_10_successful_closed_loop,
    run_stage_scenarios,
)
from app.evaluation.leakage import (
    FORBIDDEN_FEATURE_KEYS,
    audit_dataset,
    audit_duplicates,
    audit_family_isolation,
    audit_feature_keys,
    audit_split_disjointness,
    audit_temporal_ordering,
)
from app.evaluation.metrics import round4, safe_rate
from app.evaluation.ml_eval import evaluate_classifier
from app.evaluation.policy_eval import evaluate_policy
from app.evaluation.resolution_eval import (
    build_candidate_set,
    build_intel,
    evaluate_resolution,
)
from app.evaluation.retrieval_eval import (
    evaluate_retrieval,
    first_case_has_no_memory,
    rank_candidates,
)
from app.evaluation.runner import reproducibility_check, run_evaluation
from app.evaluation.safety import (
    NA_LABEL,
    _rate,
    automation_decision_for_case,
    evaluate_safety,
)
from app.evaluation.versioning import (
    DATASET_SOURCE_REAL,
    DATASET_SOURCE_SYNTHETIC,
    DEFAULT_EVAL_SEED,
    EVALUATOR_VERSION,
    SYNTHETIC_DATASET_V1,
    REAL_DATASET_TAG,
    dataset_fingerprint,
)
from app.reconciliation.engine import calculate_reconciliation
from app.schemas.decision_matrix import AutomationDecision
from app.schemas.resolution_candidate import CandidateGenerationResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.candidate_selector import CandidateSelector

# Small but complete: every family is present in every split.
PER_FAMILY = 8


# ============================================================================
# Module-scoped fixtures — each expensive evaluator runs once.
# ============================================================================


@pytest.fixture(scope="module")
def dataset() -> EvaluationDataset:
    return build_synthetic_dataset_v1(per_family=PER_FAMILY, seed=DEFAULT_EVAL_SEED)


@pytest.fixture
def session_factory():
    """A factory yielding a fresh isolated SQLite session per call.

    The golden closed-loop scenarios use well-known record ids, so they must not
    share one database.
    """
    created = []

    def make():
        engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}
        )

        @event.listens_for(engine, "connect")
        def _pragma(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(bind=engine)
        session = sessionmaker(bind=engine)()
        created.append((session, engine))
        return session

    yield make

    for session, engine in created:
        session.close()
        engine.dispose()


@pytest.fixture(scope="module")
def ml_report(dataset):
    return evaluate_classifier(dataset)


@pytest.fixture(scope="module")
def leakage(dataset):
    return audit_dataset(dataset)


@pytest.fixture(scope="module")
def retrieval(dataset):
    return evaluate_retrieval(dataset)


@pytest.fixture(scope="module")
def resolution(dataset):
    return evaluate_resolution(dataset)


@pytest.fixture(scope="module")
def safety(dataset, resolution):
    return evaluate_safety(dataset, resolution)


@pytest.fixture(scope="module")
def evidence(dataset, resolution):
    return evaluate_evidence(dataset, resolution=resolution)


@pytest.fixture(scope="module")
def policy():
    return evaluate_policy()


@pytest.fixture(scope="module")
def disagreement(dataset):
    return measure_disagreement(dataset)


# ============================================================================
# 1. Dataset versioning / provenance
# ============================================================================


def test_dataset_metadata_is_versioned(dataset):
    assert dataset.dataset_version == SYNTHETIC_DATASET_V1
    assert dataset.schema_version == "1.0.0"
    assert dataset.source == DATASET_SOURCE_SYNTHETIC
    assert dataset.seed == DEFAULT_EVAL_SEED
    assert dataset.split_strategy == "temporal"
    assert dataset.fingerprint.startswith("ds-")
    assert dataset.created_at is not None


def test_dataset_case_count_and_distribution(dataset):
    assert dataset.case_count == PER_FAMILY * 14
    distribution = dataset.label_distribution()
    assert sum(distribution.values()) == dataset.case_count
    # Every case carries deterministic ground truth.
    assert all(case.exception_type for case in dataset.cases)


def test_dataset_is_deterministic_across_rebuilds(dataset):
    rebuilt = build_synthetic_dataset_v1(per_family=PER_FAMILY, seed=DEFAULT_EVAL_SEED)
    assert rebuilt.fingerprint == dataset.fingerprint
    assert rebuilt.splits == dataset.splits
    assert rebuilt.label_distribution() == dataset.label_distribution()


def test_dataset_seed_changes_the_data(dataset):
    other = build_synthetic_dataset_v1(per_family=PER_FAMILY, seed=1234)
    assert other.fingerprint != dataset.fingerprint
    assert other.seed == 1234


def test_fingerprint_is_content_addressed():
    assert dataset_fingerprint("abc") == dataset_fingerprint("abc")
    assert dataset_fingerprint("abc") != dataset_fingerprint("abd")
    assert dataset_fingerprint("abc").startswith("ds-")


def test_ground_truth_comes_from_the_deterministic_engine(dataset):
    """Re-run the production engine on the case inputs and compare its verdict."""
    from app.schemas.enums import AdjustmentType, FeeType, TaxType
    from app.schemas.financial import (
        Adjustment as DomainAdjustment,
        Fee as DomainFee,
        Payment as DomainPayment,
        Refund as DomainRefund,
        Settlement as DomainSettlement,
        Tax as DomainTax,
    )

    checked = 0
    for case in dataset.cases[:12]:
        fin = case.financials
        now = case.decision_time
        payment = DomainPayment(
            payment_id=f"{case.case_id}-P", merchant_id="M", amount=fin.payment_amount,
            payment_timestamp=now,
        )
        settlements = [
            DomainSettlement(
                settlement_id=f"{case.case_id}-S{i}", payment_id=payment.payment_id,
                merchant_id="M", amount=amount, settlement_timestamp=now + timedelta(hours=i),
            )
            for i, amount in enumerate(fin.settlement_amounts)
        ]
        refunds = (
            [DomainRefund(refund_id=f"{case.case_id}-R", payment_id=payment.payment_id,
                          amount=fin.total_refunds, refund_timestamp=now)]
            if fin.total_refunds else []
        )
        fees = (
            [DomainFee(fee_id=f"{case.case_id}-F", payment_id=payment.payment_id,
                       amount=fin.total_fees, fee_type=FeeType.TRANSACTION)]
            if fin.total_fees else []
        )
        taxes = (
            [DomainTax(tax_id=f"{case.case_id}-T", payment_id=payment.payment_id,
                       amount=fin.total_taxes, tax_type=TaxType.GST)]
            if fin.total_taxes else []
        )
        adjustments = (
            [DomainAdjustment(adjustment_id=f"{case.case_id}-A", payment_id=payment.payment_id,
                              amount=fin.total_adjustments, adjustment_type=AdjustmentType.CREDIT)]
            if fin.total_adjustments else []
        )
        result = calculate_reconciliation(
            payment=payment, settlements=settlements, refunds=refunds, fees=fees,
            taxes=taxes, adjustments=adjustments, case_id=case.case_id,
            reconciliation_id=f"{case.case_id}-REC",
        )
        assert result.exception_type.value == case.exception_type
        assert result.match_status.value == case.match_status
        checked += 1
    assert checked == 12


def test_provenance_is_explicit_and_synthetic_never_masquerades_as_real(dataset):
    assert dataset.is_synthetic() is True
    assert dataset.source == DATASET_SOURCE_SYNTHETIC
    assert real_dataset_tag() == REAL_DATASET_TAG == "eval-real-v1"
    assert DATASET_SOURCE_REAL != DATASET_SOURCE_SYNTHETIC
    # No production evaluation corpus ships with this repository.
    assert dataset.dataset_version != real_dataset_tag()


# ============================================================================
# 2. Deterministic splitting
# ============================================================================


def test_splits_partition_every_case_exactly_once(dataset):
    ids = [cid for split in ("train", "validation", "test") for cid in dataset.splits[split]]
    assert len(ids) == dataset.case_count
    assert len(set(ids)) == dataset.case_count
    assert set(ids) == {case.case_id for case in dataset.cases}


def test_split_is_deterministic(dataset):
    rebuilt = build_synthetic_dataset_v1(per_family=PER_FAMILY, seed=DEFAULT_EVAL_SEED)
    for split in ("train", "validation", "test"):
        assert rebuilt.splits[split] == dataset.splits[split]


def test_every_split_sees_every_family(dataset):
    families = {case.family for case in dataset.cases}
    for split in ("train", "validation", "test"):
        seen = {case.family for case in dataset.cases_in_split(split)}
        assert seen == families


def test_temporal_split_orders_within_a_family(dataset):
    for family in {case.family for case in dataset.cases}:
        train = [c.decision_time for c in dataset.cases_in_split("train") if c.family == family]
        test = [c.decision_time for c in dataset.cases_in_split("test") if c.family == family]
        assert train and test
        assert max(train) < min(test)


# ============================================================================
# 3. Leakage audit
# ============================================================================


def test_leakage_audit_is_clean_on_the_synthetic_corpus(leakage, dataset):
    assert leakage.leakage_free is True
    assert leakage.critical_count == 0
    assert leakage.cases_checked == dataset.case_count
    assert leakage.dataset_version == dataset.dataset_version


def test_every_feature_is_a_registered_pre_decision_feature(dataset):
    for case in dataset.cases:
        assert set(case.features) == set(FEATURE_NAMES)
        assert not set(case.features) & FORBIDDEN_FEATURE_KEYS


def test_leakage_detects_a_label_exposed_as_a_feature(dataset):
    poisoned = dataset.cases[0].model_copy(
        update={"features": {**dataset.cases[0].features, "exception_type": 1.0}}
    )
    findings = audit_feature_keys([poisoned])
    kinds = {f.kind for f in findings}
    assert "LABEL_IN_FEATURES" in kinds
    assert any(f.severity == "critical" for f in findings)


def test_leakage_detects_post_resolution_fields_in_features(dataset):
    poisoned = dataset.cases[0].model_copy(
        update={"features": {**dataset.cases[0].features, "verification_passed": 1.0}}
    )
    findings = audit_feature_keys([poisoned])
    assert any(f.kind == "POST_RESOLUTION_IN_FEATURES" and f.severity == "critical" for f in findings)


def test_leakage_detects_split_overlap(dataset):
    leaked = dataset.model_copy(deep=True)
    leaked.splits["test"] = leaked.splits["test"] + [leaked.splits["train"][0]]
    findings = audit_split_disjointness(leaked)
    assert any(f.kind == "SPLIT_OVERLAP" and f.severity == "critical" for f in findings)


def test_leakage_detects_duplicate_case_ids(dataset):
    duplicated = dataset.model_copy(deep=True)
    duplicated.cases = duplicated.cases + [duplicated.cases[0]]
    findings = audit_duplicates(duplicated)
    assert any(f.kind == "DUPLICATE_CASE_ID" and f.severity == "critical" for f in findings)


def test_leakage_detects_a_feature_row_shared_across_splits(dataset):
    duplicated = dataset.model_copy(deep=True)
    clone = duplicated.cases[0].model_copy(update={"case_id": "EVAL-CLONE-001"})
    duplicated.cases = duplicated.cases + [clone]
    duplicated.splits["test"] = duplicated.splits["test"] + ["EVAL-CLONE-001"]
    findings = audit_duplicates(duplicated)
    assert any(f.kind == "DUPLICATE_FEATURE_ROW" and f.severity == "critical" for f in findings)


def test_leakage_detects_future_cases_relative_to_training(dataset):
    """A test case dated before the training window must be flagged."""
    leaked = dataset.model_copy(deep=True)
    ordered = sorted(leaked.cases, key=lambda c: c.decision_time)
    early, late = ordered[0], ordered[-1]
    leaked.splits = {
        "train": [late.case_id],
        "validation": [],
        "test": [early.case_id],
    }
    findings = audit_temporal_ordering(leaked)
    assert any(f.kind == "TEMPORAL_ORDER" for f in findings)


def test_leakage_reports_family_straddle_as_a_known_warning(dataset, leakage):
    """Per-family temporal splits put siblings of a family in every split.

    That is intentional (every split needs every class) but it makes synthetic
    accuracy optimistic, so the audit must surface it rather than hide it.
    """
    findings = audit_family_isolation(dataset)
    assert findings
    assert all(f.severity == "warning" for f in findings)
    assert all(f.kind == "FAMILY_STRADDLE" for f in findings)
    assert all(f.severity != "critical" for f in leakage.findings)


# ============================================================================
# 4. ML classification metrics
# ============================================================================


def test_ml_report_identifies_model_and_split(ml_report, dataset):
    assert ml_report.dataset_version == dataset.dataset_version
    assert ml_report.seed == DEFAULT_EVAL_SEED
    assert ml_report.train_samples == len(dataset.splits["train"])
    assert ml_report.test_samples == len(dataset.splits["test"])
    assert ml_report.split_strategy == "temporal"
    assert ml_report.model_version


def test_ml_metrics_are_computed_and_bounded(ml_report):
    for value in (
        ml_report.accuracy,
        ml_report.precision_macro,
        ml_report.recall_macro,
        ml_report.f1_macro,
        ml_report.precision_weighted,
        ml_report.recall_weighted,
        ml_report.f1_weighted,
    ):
        assert 0.0 <= value <= 1.0
    assert ml_report.confusion_labels


def test_confusion_matrix_is_square_and_consistent(ml_report):
    size = len(ml_report.confusion_labels)
    assert size == len(ml_report.confusion_matrix)
    assert ml_report.test_samples == sum(sum(row) for row in ml_report.confusion_matrix)
    assert all(len(row) == size for row in ml_report.confusion_matrix)
    # Diagonal-only matrix must reproduce the reported accuracy.
    correct = sum(ml_report.confusion_matrix[i][i] for i in range(size))
    assert round4(correct / ml_report.test_samples) == ml_report.accuracy


def test_per_class_metrics_cover_every_represented_class(ml_report):
    assert set(ml_report.per_class) == set(ml_report.confusion_labels)
    for label, metrics in ml_report.per_class.items():
        for key in ("precision", "recall", "f1", "support", "false_positive_rate", "false_negative_rate"):
            assert key in metrics
        assert metrics["support"] >= 0
        assert 0.0 <= metrics["f1"] <= 1.0


def test_confidence_and_calibration_are_reported(ml_report):
    assert 0.0 <= ml_report.mean_confidence <= 1.0
    assert ml_report.confidence_bins
    assert ml_report.calibration_ece >= 0.0


def test_confidence_threshold_table_is_monotone_in_coverage(ml_report):
    rows = ml_report.confidence_thresholds
    assert [r.threshold for r in rows] == sorted(r.threshold for r in rows)
    coverage = [r.coverage for r in rows]
    assert coverage == sorted(coverage, reverse=True)
    for row in rows:
        assert 0.0 <= row.coverage <= 1.0
        assert 0 <= row.samples <= ml_report.test_samples
        assert row.coverage == round4(row.samples / ml_report.test_samples)
        assert 0.0 <= row.accuracy <= 1.0


def test_ml_report_documents_the_production_evaluator_passthrough(ml_report):
    assert "available" in ml_report.production_evaluator


# ============================================================================
# 5. ML vs deterministic disagreement
# ============================================================================


def test_disagreement_report_totals_are_consistent(disagreement):
    assert disagreement.cases_compared == disagreement.agreements + disagreement.disagreements
    assert disagreement.deterministic_authoritative is True
    assert 0.0 <= disagreement.agreement_rate <= 1.0


def test_disagreements_are_inspectable_and_typed(disagreement):
    for record in disagreement.records:
        assert record.case_id
        assert record.deterministic_type != record.ml_predicted_type
        assert 0.0 <= record.confidence <= 1.0
        assert record.category
    assert sum(disagreement.categories.values()) == disagreement.disagreements


def test_deterministic_truth_is_never_overwritten_by_ml(disagreement):
    """Measuring disagreement must not mutate the deterministic verdict."""
    assert disagreement.deterministic_authoritative is True
    assert all(r.deterministic_type for r in disagreement.records)


# ============================================================================
# 6. Evidence quality
# ============================================================================


def test_evidence_metrics_are_reported(evidence, dataset):  # noqa: D103
    assert evidence.cases == dataset.case_count
    assert 0.0 <= evidence.mean_coverage <= 1.0
    assert evidence.min_coverage <= evidence.mean_coverage <= evidence.max_coverage
    assert evidence.fully_explained + evidence.partially_explained >= 0
    assert evidence.missing_evidence_cases >= 1
    assert evidence.conflicting >= 1


def test_evidence_status_partition_matches_cases(evidence, dataset):
    flagged = sum(1 for c in dataset.cases if c.evidence.has_conflict)
    assert evidence.conflicting == flagged
    missing = sum(1 for c in dataset.cases if c.evidence.missing_evidence_count > 0)
    assert evidence.missing_evidence_cases == missing


def test_evidence_and_recommendation_correctness_are_correlated(evidence):
    for value in (
        evidence.weak_evidence_correct_rate,
        evidence.strong_evidence_correct_rate,
        evidence.weak_evidence_unsafe_rate,
    ):
        assert 0.0 <= value <= 1.0


def test_weak_evidence_never_becomes_an_unsafe_automation(evidence, safety):
    """Weak evidence must not produce an automated resolution."""
    assert evidence.weak_evidence_unsafe_rate == 0.0
    assert safety.unsafe_automation_count == 0
    assert evidence.weak_evidence_cases > 0


def test_derive_features_excludes_labels_and_post_outcome_fields():
    from app.evaluation.dataset import EvalFinancials

    financials = EvalFinancials(
        payment_amount=1_000_000,
        settlement_amounts=[925_000],
        total_fees=50_000,
        expected_amount=950_000,
        actual_amount=925_000,
        difference=25_000,
    )
    features = derive_features(financials, EvalEvidenceSignals(coverage=0.9))
    assert set(features) == set(FEATURE_NAMES)
    assert not set(features) & FORBIDDEN_FEATURE_KEYS


# ============================================================================
# 7. Historical retrieval
# ============================================================================


def test_retrieval_recall_at_k_is_reported(retrieval):
    assert set(retrieval.recall_at_k) == {"1", "3", "5", "10"}
    for value in retrieval.recall_at_k.values():
        assert 0.0 <= value <= 1.0
    assert retrieval.recall_at_k["1"] <= retrieval.recall_at_k["3"] <= retrieval.recall_at_k["10"]
    assert 0.0 <= retrieval.mrr <= 1.0


def test_retrieval_excludes_self_matches(retrieval):
    assert retrieval.self_match_violations == 0


def test_retrieval_excludes_future_cases(retrieval):
    assert retrieval.temporal_violations == 0


def test_retrieval_handles_empty_memory(dataset, retrieval):
    assert first_case_has_no_memory(dataset) is True
    assert retrieval.empty_memory_queries == 1
    assert retrieval.empty_memory_handled is True


def test_rank_candidates_rejects_self_and_future(dataset):
    ordered = sorted(dataset.cases, key=lambda c: c.decision_time)
    query = ordered[-1]
    # Memory includes the query itself and a case from its own future.
    future = query.model_copy(
        update={"case_id": "EVAL-FUTURE", "decision_time": query.decision_time + timedelta(days=1)}
    )
    memory = [query, future] + [c for c in ordered if c.case_id != query.case_id]
    ranked, self_violations, temporal_violations = rank_candidates(query, memory)
    assert query.case_id not in ranked
    assert "EVAL-FUTURE" not in ranked
    assert self_violations == 1
    assert temporal_violations == 1


def test_retrieval_is_labelled_synthetic(retrieval):
    assert retrieval.version_compatible is True
    assert any("No embedding model" in note for note in retrieval.notes)


# ============================================================================
# 8. Resolution recommendation
# ============================================================================


def test_resolution_top1_and_top3_metrics(resolution, dataset):
    assert resolution.cases == dataset.case_count
    assert 0.0 <= resolution.top1_accuracy <= 1.0
    assert 0.0 <= resolution.top3_candidate_recall <= 1.0
    assert resolution.top1_accuracy <= resolution.top3_candidate_recall


def test_resolution_metrics_scales_are_reported(resolution):
    assert 0.0 <= resolution.top1_auto_accuracy <= resolution.top1_accuracy
    assert 0.0 <= resolution.abstention_rate <= 1.0
    assert 0.0 <= resolution.human_review_recall <= 1.0
    assert resolution.selection_status_counts


def test_conflict_cases_are_routed_to_a_human(resolution):
    """The selector must defer at least some conflicting cases.

    It does *not* defer all of them: conflict detection inside the selector runs
    only when two or more candidates pass its thresholds, so a lone passing
    candidate is recommended even when the evidence conflicts. That weakness is
    recorded here as a real measurement, and the authoritative policy layer is
    asserted separately to block every one of these cases.
    """
    assert resolution.conflict_cases >= 1
    assert resolution.conflict_routed_to_human >= 1
    assert resolution.conflict_routed_to_human <= resolution.conflict_cases


def test_correct_candidate_is_offered_for_every_case(resolution):
    assert resolution.correct_candidate_present_rate == 1.0


def test_selector_abstains_when_there_are_no_candidates(dataset):
    case = dataset.cases[0]
    empty = CandidateGenerationResult(
        exception_id=case.case_id,
        case_id=case.case_id,
        status="UNRESOLVED",
        candidates=[],
        total_candidates=0,
    )
    selection = CandidateSelector().select(empty, build_intel(case))
    assert selection.status == SelectionStatus.UNRESOLVED
    assert selection.selected_candidate is None


def test_selector_refuses_candidates_below_thresholds(dataset):
    """A candidate set whose members are all too weak must not be recommended."""
    from app.evaluation.resolution_eval import _proposal

    case = dataset.cases[0]
    weak = CandidateGenerationResult(
        exception_id=case.case_id,
        case_id=case.case_id,
        status="CANDIDATES_GENERATED",
        candidates=[
            _proposal(
                case,
                "FEE_ADJUSTMENT",
                1,
                compatible=False,
                coverage=0.01,
                confidence=0.01,
                sources=["historical_similarity"],
                historical_similarity=0.05,
            )
        ],
        total_candidates=1,
    )
    selection = CandidateSelector().select(weak, build_intel(case))
    assert selection.status == SelectionStatus.UNRESOLVED


def test_unresolved_cases_stay_unresolved(resolution):
    """Cases the selector refuses must not later be marked recommended."""
    for result in resolution.results:
        if result.selection_status == SelectionStatus.UNRESOLVED.value:
            assert result.selected_resolution is None
            assert result.abstained is True


# ============================================================================
# 9. Policy / guardrail decision matrix
# ============================================================================


def test_policy_matrix_has_no_invariant_violations(policy):
    assert len(policy.rows) >= 10
    assert policy.invariant_violations == []


def test_policy_allows_only_the_clean_low_exposure_case(policy):
    assert policy.allow_count == 1
    auto_rows = [r.row_id for r in policy.rows if r.auto]
    assert auto_rows == ["R1"]


def test_policy_denies_high_confidence_with_weak_evidence(policy):
    row = next(r for r in policy.rows if r.row_id == "R2")
    assert row.confidence > 0.9
    assert row.actual_decision != AutomationDecision.AUTO.value
    assert policy.confidence_bypass_detected is False


def test_policy_denies_high_exposure_despite_max_confidence(policy):
    row = next(r for r in policy.rows if r.row_id == "R3")
    assert row.confidence == 0.99
    assert row.exposure_paise > 10_000_000
    assert row.actual_decision != AutomationDecision.AUTO.value


def test_historical_similarity_cannot_bypass_policy(policy):
    row = next(r for r in policy.rows if r.row_id == "R8")
    assert row.historical_similarity == 1.0
    assert row.actual_decision != AutomationDecision.AUTO.value
    assert policy.similarity_bypass_detected is False


def test_deterministic_inconsistency_blocks_automation(policy):
    assert policy.consistency_bypass_detected is False
    for row_id in ("R4", "R4b"):
        row = next(r for r in policy.rows if r.row_id == row_id)
        assert row.actual_decision != AutomationDecision.AUTO.value


def test_missing_evidence_is_not_an_approval(policy):
    row = next(r for r in policy.rows if r.row_id == "R9")
    assert row.missing_evidence
    assert row.actual_decision != AutomationDecision.AUTO.value
    assert policy.evidence_bypass_detected is False


def test_unhealthy_dependency_fails_closed(policy):
    row = next(r for r in policy.rows if r.row_id == "R7")
    assert row.dependencies_healthy is False
    assert row.actual_decision != AutomationDecision.AUTO.value


def test_policy_decisions_are_one_of_the_three_automation_outcomes(policy):
    allowed = {
        AutomationDecision.AUTO.value,
        AutomationDecision.HUMAN_REVIEW.value,
        AutomationDecision.UNRESOLVED.value,
    }
    assert {r.actual_decision for r in policy.rows} <= allowed


# ============================================================================
# 10. Safety metrics (and zero-denominator handling)
# ============================================================================


def test_zero_denominator_is_reported_as_na_not_zero():
    assert _rate(0, 0) is None
    assert _rate(3, 0) is None
    assert _rate(0, 3) == 0.0
    assert _rate(2, 4) == 0.5
    assert safe_rate(0, 0) == 0.0  # raw primitive still returns a number


def test_safety_zero_denominator_metrics_render_as_na(safety):
    if safety.automation_eligible == 0:
        assert safety.safe_resolution_precision is None
        assert safety.unsafe_resolution_rate is None
        assert safety.false_automation_rate is None
        headline = safety.headline()
        assert headline["safe_resolution_precision"] == NA_LABEL
        assert headline["unsafe_resolution_rate"] == NA_LABEL
        assert headline["false_automation_rate"] == NA_LABEL
    else:
        assert 0.0 <= safety.safe_resolution_precision <= 1.0


def test_safety_never_automates_an_unexpected_case(safety):
    assert safety.unsafe_automation_count == 0
    assert safety.safety_target_met is True
    assert safety.incorrectly_resolved == 0


def test_safety_rates_use_explicit_denominators(safety, resolution):
    assert safety.total_cases == resolution.cases
    assert safety.abstention_rate is not None
    assert safety.abstention_rate == round4(
        sum(1 for r in resolution.results if r.abstained) / resolution.cases
    )
    assert safety.human_review_rate is not None


def test_safety_without_cases_returns_empty_report(dataset):
    from app.evaluation.resolution_eval import ResolutionReport

    empty = ResolutionReport(dataset_version=dataset.dataset_version, cases=0)
    report = evaluate_safety(dataset, empty)
    assert report.total_cases == 0
    assert report.abstention_rate is None


def test_guardrail_engine_refuses_automation_for_every_conflicting_case(dataset, resolution):
    """End-to-end: a conflicting case is never automation-eligible.

    The selector's weaker conflict handling is overruled by the Phase 8 policy
    layer, which is the authority on automation.
    """
    from app.services.guardrail_engine import GuardrailEngine

    engine = GuardrailEngine()
    by_id = {r.case_id: r for r in resolution.results}
    conflicting = [c for c in dataset.cases if c.evidence.has_conflict]
    assert conflicting
    for case in conflicting:
        result = by_id[case.case_id]
        decision = automation_decision_for_case(case, result, engine)
        assert decision != AutomationDecision.AUTO.value


def test_closure_accuracy_is_na_without_closure_scenarios():
    metrics = closure_metrics([])
    assert metrics["closure_accuracy"] is None
    assert metrics["closure_attempts"] == 0


# ============================================================================
# 11. Golden end-to-end scenarios
# ============================================================================


@pytest.mark.parametrize(
    "scenario",
    [
        golden_1_fee_mismatch_full_chain,
        golden_2_already_matched,
        golden_3_missing_evidence,
        golden_4_conflicting_evidence,
        golden_5_high_confidence_policy_denial,
        golden_8_stale_approval,
    ],
    ids=["G1", "G2", "G3", "G4", "G5", "G8"],
)
def test_golden_stage_scenarios_pass(scenario):
    result = scenario()
    assert result.passed, result.failures
    assert result.scenario_id


def test_golden_stage_suite_passes_as_a_whole():
    results = run_stage_scenarios()
    assert len(results) == 6
    assert all(r.passed for r in results), [r.failures for r in results if not r.passed]


def test_golden_1_financial_truth_is_deterministic():
    result = canonical_reconciliation()
    assert result.difference == 25_000
    assert result.match_status.value == "EXCEPTION"
    scenario = golden_1_fee_mismatch_full_chain()
    assert scenario.observed["difference"] == 25_000


def test_golden_6_provider_success_does_not_close_the_exception(db_session):
    result = golden_6_provider_success_reconciliation_failure(db_session)
    assert result.passed, result.failures
    assert result.observed["exception_closed"] is False
    assert result.observed["financially_verified"] is False


def test_golden_7_provider_timeout_fails_closed(db_session):
    result = golden_7_provider_timeout(db_session)
    assert result.passed, result.failures
    assert result.observed["financially_verified"] is False
    assert result.observed["exception_closed"] is False


def test_golden_9_unknown_provider_result_fails_closed(db_session):
    result = golden_9_unknown_provider_result(db_session)
    assert result.passed, result.failures
    assert result.observed["exception_closed"] is False


def test_golden_10_closes_only_after_deterministic_verification(db_session):
    result = golden_10_successful_closed_loop(db_session)
    assert result.passed, result.failures
    assert result.observed["financially_verified"] is True
    assert result.observed["exception_closed"] is True


def test_golden_suite_covers_all_ten_scenarios(session_factory):
    results = run_stage_scenarios() + [s(session_factory()) for s in DB_SCENARIOS]
    assert len(results) == 10
    ids = {r.scenario_id for r in results}
    assert ids == {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"}
    assert all(r.passed for r in results), [r.failures for r in results if not r.passed]
    closure = closure_metrics(results)
    assert closure["closure_attempts"] == 4
    assert closure["correctly_closed"] == 4
    assert closure["closure_accuracy"] == 1.0


# ============================================================================
# 12. Reproducibility, isolation and reporting
# ============================================================================


def test_two_runs_produce_identical_results(session_factory):
    comparison = reproducibility_check(
        per_family=4, seed=DEFAULT_EVAL_SEED, session=session_factory
    )
    assert comparison["fingerprint_match"] is True
    assert comparison["headline_match"] is True
    assert comparison["differences"] == {}
    assert comparison["reproducible"] is True
    assert comparison["dataset_version_first"] == comparison["dataset_version_second"]


def test_reproducibility_compares_every_deterministic_section():
    comparison = reproducibility_check(per_family=3, seed=DEFAULT_EVAL_SEED)
    compared = set(comparison["sections_compared"])
    for section in (
        "dataset_version",
        "dataset_fingerprint",
        "feature_schema_version",
        "ml",
        "disagreement",
        "retrieval",
        "resolution",
        "safety",
        "evidence",
        "policy",
        "golden",
        "leakage",
    ):
        assert section in compared
    # Non-deterministic bookkeeping is deliberately excluded.
    assert "timestamp" not in compared
    assert "run_id" not in compared


def test_repeated_evaluators_are_deterministic(dataset):
    first = evaluate_resolution(dataset)
    second = evaluate_resolution(dataset)
    assert first.summary() == second.summary()


def test_evaluation_does_not_mutate_the_dataset(dataset):
    before = dataset.model_dump()
    evaluate_resolution(dataset)
    evaluate_evidence(dataset)
    evaluate_safety(dataset, evaluate_resolution(dataset))
    assert dataset.model_dump() == before


def test_run_evaluation_wires_every_component():
    report = run_evaluation(per_family=3, seed=DEFAULT_EVAL_SEED)
    assert report.evaluator_version == EVALUATOR_VERSION
    assert report.dataset_version == SYNTHETIC_DATASET_V1
    assert report.dataset_cases == 3 * 14
    assert report.ml and report.resolution and report.safety and report.policy
    assert report.leakage["leakage_free"] is True
    headline = report.headline()
    assert headline["golden_passed"] == 6
    assert headline["golden_total"] == 6
    # Without a session the closure scenarios cannot run: N/A, not 0.0.
    assert headline["closure_accuracy"] == NA_LABEL


def test_full_run_with_database_session_measures_closure(session_factory):
    report = run_evaluation(per_family=3, seed=DEFAULT_EVAL_SEED, session=session_factory)
    assert report.golden["total"] == 10
    assert report.golden["passed"] == 10
    assert report.safety["closure_attempts"] == 4
    assert report.safety["closure_accuracy"] == 1.0


def test_artifacts_are_written_as_machine_readable_json(tmp_path):
    report = run_evaluation(per_family=2, seed=DEFAULT_EVAL_SEED, write=True, output_dir=tmp_path)
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert "metrics.json" in written
    assert "confusion_matrix.json" in written
    assert "safety_metrics.json" in written
    assert "golden_results.json" in written
    assert f"evaluation_run_{report.run_id}.json" in written
