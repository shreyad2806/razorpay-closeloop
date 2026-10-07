# Phase 15 — Evaluation & Quality Measurement

Phase 15 answers one question: **does CloseLoop's intelligence work correctly and
safely?** It is *measurement only*. No production decision logic, threshold,
weight, prompt, feature or state machine was changed to make these numbers look
better.

> **Every number in this document is fixture performance on a synthetic corpus,
> not production performance.** No production evaluation corpus exists in this
> repository.

---

## 1. Layout

| Path | Responsibility |
| --- | --- |
| `app/evaluation/versioning.py` | evaluator / dataset version pins, dataset fingerprint |
| `app/evaluation/metrics.py` | pure metric primitives (confusion, PR/F1, ECE, Recall@K, rates) |
| `app/evaluation/dataset.py` | versioned synthetic dataset; ground truth from the **real** reconciliation engine |
| `app/evaluation/leakage.py` | leakage audit (labels, post-outcome fields, split overlap, duplicates, temporal order, family isolation) |
| `app/evaluation/ml_eval.py` | production XGBoost classifier metrics, per-class, calibration, selective prediction |
| `app/evaluation/disagreement.py` | ML vs deterministic disagreement, categorised |
| `app/evaluation/evidence_eval.py` | evidence coverage / consistency / missing / conflict quality |
| `app/evaluation/retrieval_eval.py` | historical-memory retrieval contract (Recall@K, self-match, future-case) |
| `app/evaluation/resolution_eval.py` | real candidate selector: Top-1 / Top-3 / abstention / human review |
| `app/evaluation/policy_eval.py` | guardrail decision matrix + bypass invariants |
| `app/evaluation/safety.py` | automation safety metrics with explicit denominators |
| `app/evaluation/golden.py` | 10 deterministic end-to-end golden scenarios |
| `app/evaluation/runner.py` | orchestration, headline, reproducibility check |
| `app/evaluation/artifacts.py` | machine-readable JSON artifacts |
| `scripts/run_phase15_evaluation.py` | CLI entry point |
| `tests/test_phase15_evaluation.py` | 84 executable behaviour tests |

Everything the harness measures runs the **production** component: the
reconciliation engine produces the ground truth, the production classifier is
fitted, the production selector chooses candidates, the production guardrail
engine authorises automation, and the production execution / post-execution
reconciliation services decide closure.

## 2. Running it

```bash
cd backend

# Phase 15 tests
python -m pytest tests/test_phase15_evaluation.py -q -p no:warnings

# Full evaluation + artifacts (isolated in-memory SQLite, nothing real is touched)
python -m scripts.run_phase15_evaluation --per-family 20
```

Artifacts are written to `backend/evaluation_results/` (`metrics.json`,
`confusion_matrix.json`, `resolution_metrics.json`, `safety_metrics.json`,
`retrieval_metrics.json`, `golden_results.json`, `leakage_report.json`,
`policy_report.json`, `evidence_metrics.json`, and the full run JSON). They are
generated output and are not meant to be committed.

## 3. Dataset provenance

| Dataset | Version | Source | Notes |
| --- | --- | --- | --- |
| Synthetic evaluation corpus | `eval-synthetic-v1` (schema `1.0.0`) | `SYNTHETIC` | 14 families × N cases, seed 42, fingerprint `ds-30b95d8a82792087` at `per_family=20` |
| Golden closed-loop scenarios | in `golden.py` | fixture | 10 deterministic scenarios, canonical FEE_MISMATCH paise |
| Real historical corpus | — | `REAL_HISTORICAL` | **Not available.** No production evaluation corpus ships with this repository. |

Synthetic and real provenance are separate enum values and can never be silently
combined; there is no code path that merges them. Labels are never hand-written:
each case's `exception_type` / `match_status` / `difference` is whatever
`app.reconciliation.engine.calculate_reconciliation` returns for that case's
inputs.

Splits are per-family temporal (`train` 60% / `validation` 20% / `test` 20%)
so every split sees every class while ordering within a family stays temporal.

## 4. Metric definitions

| Metric | Formula |
| --- | --- |
| Safe Resolution Precision | correctly verified automated resolutions / all automated resolutions |
| Unsafe Resolution Rate | automated resolutions that should not have been automated / all automated resolutions |
| False Automation Rate | guardrail-invalid automated resolutions / all automated resolutions |
| Human Review Rate | cases deferred to human / all evaluated exceptions |
| Human Review Recall | cases requiring human review that were deferred / cases requiring human review |
| Abstention Rate | (human review + unresolved) / all evaluated exceptions |
| Verification Failure Rate | closure decisions that contradict the expected outcome / closure decisions attempted |
| Closure Accuracy | closure decisions matching the expected outcome / closure decisions attempted |
| Closure Precision | correctly closed / all closed |

**Zero-denominator rule.** When a denominator is zero the metric is reported as
`None` internally and rendered as **`N/A — no applicable cases`**. It is never
reported as `0.0` (which would imply measured perfection) or `1.0` (which would
imply measured safety). This is enforced by `_rate()` in
`app/evaluation/safety.py` and asserted by the test suite.

## 5. Measured snapshot

`per_family=20`, `seed=42`, 280 cases, fingerprint `ds-30b95d8a82792087`.

### ML (production XGBoost classifier)

Trained on 168 cases, evaluated on 56 test cases (9 classes).

accuracy `0.9821` · macro P `0.9778` · macro R `0.9931` · macro F1 `0.9841` ·
weighted F1 `0.9828` · macro FPR `0.0021` · macro FNR `0.0069` ·
mean confidence `0.9388` · ECE `0.0434`

Lowest per-class F1: `EXACT_MATCH` 0.8889 (precision 0.80, recall 1.00), then
`FEE_MISMATCH` 0.9677 (precision 1.00, recall 0.9375). Every other class scores
1.0. `UNKNOWN` is neither over- nor under-predicted (8 actual / 8 predicted,
precision and recall 1.0).

Selective prediction is flat: coverage `0.9643` at threshold 0.5, `0.9464` from
0.6 upward, with accuracy `0.9815` → `0.9811`. Almost all errors sit in the top
confidence bin, so **a higher confidence threshold does not buy safety on this
corpus**.

### Deterministic vs ML

Agreement `0.9821` (55/56). One disagreement: `EVAL-FEE_MISMATCH-017`,
deterministic `FEE_MISMATCH` vs ML `EXACT_MATCH` at confidence 0.9747 — a
high-confidence disagreement, which is exactly the case the deterministic engine
must win. Deterministic remains authoritative.

### Evidence

mean coverage `0.719` (min 0.103, max 0.998) · mean consistency `0.6786` ·
fully explained 100 / partially 80 / unsupported 100 · conflicting 60 ·
missing-evidence cases 80 · weak-evidence cases 200.

Weak-evidence recommendation correctness `0.51` vs strong-evidence `0.2889`
(the correctness measure is "selector recommended *and* correct *and* the case
was expected to be automated", which is dominated by how often a case was
expected to be automated — see the limitation in §6). Weak-evidence unsafe
automations: **0**.

### Historical retrieval (synthetic mechanics only, no embedding model)

Recall@1 `0.1695` · @3 `0.3917` · @5 `0.5459` · @10 `0.7796` · MRR `0.9317` ·
Precision@1 `0.9176` · queries 279 · self-match violations `0` · future-case
violations `0` · empty-memory queries `1` (handled).

Recall@1 is low because relevance is "same deterministic family" and families
deliberately overlap in feature space. These numbers describe the retrieval
*contract*, not production ranking quality.

### Resolution

Top-1 accuracy `0.825` · Top-1 auto accuracy `0.3679` · Top-3 candidate recall
`1.0` · correct candidate present `1.0` · human-review recall `0.485` ·
abstention `0.175` · unresolved-when-no-safe-candidate `0.23` ·
status counts `HUMAN_REVIEW 128 / RECOMMENDED 103 / UNRESOLVED 49` ·
conflict cases 60, of which 34 were deferred by the selector.

**Finding.** The selector is not conflict-averse when only one candidate passes
its thresholds: the conflict checks in `CandidateSelector` run only when two or
more candidates pass, so 26 conflicting-evidence cases were still recommended.
The Phase 8 guardrail blocked every one of them (see §5 policy and the safety
assertions), so no unsafe automation results — but the selector's own conflict
handling is weaker than the policy layer's and should be reviewed.

### Policy / guardrail matrix

10 rows, 1 `AUTO` / 5 `HUMAN_REVIEW` / 4 `UNRESOLVED`, **0 invariant
violations**, and all four bypass detectors false:

* ML confidence cannot bypass policy (high-confidence + weak evidence → not AUTO)
* Historical similarity cannot bypass policy (similarity 1.0 + weak evidence → not AUTO)
* Missing evidence is not approval
* Deterministic inconsistency / explicit conflict blocks automation
* Unhealthy critical dependency fails closed
* Very high exposure blocks automation regardless of confidence

### Safety

| Metric | Value |
| --- | --- |
| Automation eligible | `0` |
| Unsafe automation count | `0` |
| Unsafe Resolution Rate | `N/A — no applicable cases` |
| Safe Resolution Precision | `N/A — no applicable cases` |
| False Automation Rate | `N/A — no applicable cases` |
| Human Review Rate | `0.5` |
| Human Review Recall | `1.0` |
| Abstention Rate | `0.175` |
| Verification Failure Rate | `0.0` (0 of 4 closure decisions) |
| Closure Accuracy | `1.0` (4 of 4 closure decisions) |
| Closure Precision | `1.0` (1 of 1 closed) |

**Finding (corpus coverage).** No synthetic case survives *both* the selector
and the guardrail, so the automation-path metrics have no denominator. The
mechanical reason is threshold alignment, not a defect in this harness: the
guardrail requires confidence ≥ 0.75 and `LOW` risk for `AUTO`, the selector's
confidence for these fixtures peaks around 0.5, and `_assess_risk` assigns
`MEDIUM` above 10 000 paise of adjustment. `AUTO` is therefore unreachable on
this corpus. Safety of the automation gate is instead demonstrated by the
policy matrix (10 rows) and the golden scenarios.

### Golden closed-loop scenarios — 10/10 pass

| ID | Scenario | Outcome asserted |
| --- | --- | --- |
| G1 | FEE_MISMATCH full chain | engine says FEE_MISMATCH / difference 25 000; execution success is never closure |
| G2 | already matched | no financial exception, no resolution |
| G3 | missing evidence | no unsafe automatic resolution |
| G4 | conflicting evidence | human review / escalation |
| G5 | high ML confidence + policy denial | not executed |
| G6 | provider success + reconciliation mismatch | verification fails, exception NOT closed, escalated |
| G7 | provider read failure (timeout) | fails closed — not verified, not closed |
| G8 | stale approval | execution denied |
| G9 | non-success execution result | not verified, not closed |
| G10 | valid resolution → fresh state → deterministic reconciliation | verified and closed |

The scenarios assert the closed-loop contract: financial truth comes from
deterministic reconciliation, `after_state` from the execution result is never
trusted as closure truth, fresh provider state is used, and closure happens only
on `MATCHED` with zero discrepancy.

### Leakage

`leakage_free = true`, 0 critical findings, 14 warnings — all
`FAMILY_STRADDLE`, which is inherent to the per-family split design (see §6).

## 6. Known limitations

1. **No production evaluation corpus.** No real historical cases are available,
   so no metric here is production performance.
2. **Synthetic-only metrics.** ML accuracy, retrieval Recall@K, resolution
   accuracy and every safety rate are fixture numbers.
3. **Family straddle inflates ML accuracy.** Per-family splitting puts
   near-identical siblings of a test case in train, so the 0.9821 accuracy is
   optimistic relative to production.
4. **Zero-denominator metrics** are reported as `N/A` and are *not* measured
   results: Safe Resolution Precision, Unsafe Resolution Rate, False Automation
   Rate.
5. **No embedding model.** Retrieval is cosine over deterministic features;
   embedding-version compatibility is asserted but not exercised against a real
   encoder.
6. **No real provider integration.** Provider behaviour is the deterministic
   `MockProvider`; a real Razorpay integration is untested here.
7. **SQLite only.** No Postgres/pgvector in this environment, so database-backed
   scenarios run on SQLite.
8. **Synthetic rewards/outcomes** are frozen fixture signals, not observed
   production outcomes.
9. **Selector conflict weakness** documented in §5 is a production-behaviour
   finding. It was **not** fixed — Phase 15 records it and stops.

## 7. Protected surfaces

Confirmed untouched by Phase 15: reconciliation arithmetic and state machine,
evidence semantics, production XGBoost model/features/hyperparameters,
historical-memory retrieval semantics, candidate scoring, candidate selection,
policy/guardrail semantics, provider execution semantics, post-execution
verification semantics, LangGraph graph, MCP tools/permissions, auth/RBAC,
frontend, and the Phase 14 observability layer.

No AWS, CloudWatch, X-Ray, MLflow server, Redis, Kafka, Celery, Prometheus,
new database, dashboard or new provider integration was added.
