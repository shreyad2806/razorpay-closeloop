# Phase 14 — Backend Testing Strategy

## Codebase Audit Summary

| Metric | Value |
|--------|-------|
| Source files (app/) | 155 |
| Test files (tests/) | 87 |
| Total tests | 3562 |
| Passing tests | 3562 |
| Failing tests | 0 |
| conftest.py | Not present |
| pytest.ini / pyproject.toml | Not present |
| db_test_helper.py | SQLite in-memory helper exists |

---

## 1. Testing Pyramid

```
                    ┌──────────────┐
                    │   E2E (5)    │   Full workflow: data → reconciliation → exception → ML → guardrails → resolution → verification
                   ┌┴──────────────┴┐
                   │ Integration (30) │   Service chains: guardrail+execution, feedback+reward, candidate+scorer+selector
                  ┌┴──────────────────┴┐
                  │   Component (150)    │   Individual services: GuardrailEngine, RewardEngine, EvidenceGraphBuilder
                 ┌┴──────────────────────┴┐
                 │      Unit (3000+)        │   Pure functions, schemas, validators, calculations
                 └──────────────────────────┘
```

**Target distribution**: 80% unit, 15% component, 4% integration, 1% E2E

---

## 2. Unit Test Strategy

### 2A. Pure In-Memory Services (29 files — highest priority)

These services have zero external dependencies and are the easiest to test thoroughly:

| Service | Key Methods | Test Priority |
|---------|------------|---------------|
| `guardrail_engine.py` | `evaluate()` | P0 — financial safety |
| `reward_engine.py` | `calculate_reward()` | P0 — learning correctness |
| `feedback.py` | `record_feedback()`, `record_outcome()` | P0 — data integrity |
| `candidate_generator.py` | `generate()` | P1 — resolution quality |
| `candidate_scorer.py` | `score_candidate()`, `score_and_rank()` | P1 — scoring accuracy |
| `candidate_selector.py` | `select()` | P1 — selection logic |
| `confidence_gate.py` | `evaluate()` | P0 — safety boundary |
| `decision_matrix.py` | `evaluate()` | P0 — safety boundary |
| `evidence_graph.py` | `build()`, `get_conflicts()` | P1 — evidence correctness |
| `evidence_guard.py` | `evaluate()` | P0 — safety boundary |
| `evidence_quality.py` | `score()` | P1 — quality assessment |
| `execution.py` | `execute()`, `transition_status()` | P0 — execution safety |
| `resolution_verification.py` | `verify()` | P0 — verification correctness |
| `verification.py` | `verify()` | P0 — verification correctness |
| `financial_diff.py` | `compare()` | P0 — financial accuracy |
| `exposure_guard.py` | `evaluate()` | P0 — financial safety |
| `fallback_guard.py` | `evaluate()` | P1 — safety fallback |
| `learning_metrics.py` | `compute()`, `compare()` | P1 — metrics accuracy |
| `learning_dataset.py` | `build_dataset()`, `check_quality()` | P1 — data quality |
| `batch_learning.py` | `calculate()`, `compare()` | P1 — batch comparison |
| `self_learning_loop.py` | `record_outcome()`, `train_candidate()` | P1 — learning pipeline |
| `policy_learning.py` | `create_policy()`, `compare()` | P1 — policy versioning |
| `historical_case_store.py` | `store()`, `search()` | P2 — case retrieval |
| `idempotency.py` | `check_idempotency()`, `claim_key()` | P1 — write safety |
| `model_promotion.py` | `promote_to_production()` | P0 — model safety |
| `model_training.py` | `train()`, `evaluate()` | P1 — training correctness |
| `resolution_engine.py` | `resolve()` | P0 — resolution correctness |
| `resolution_verification.py` | `verify()` | P0 — verification |
| `rollback.py` | rollback operations | P1 — recovery |

**Test approach**: Instantiate service directly, pass synthetic data, assert outputs. No mocks needed.

### 2B. Schemas (48 files — extensive coverage needed)

All Pydantic schemas should have:
- Valid input acceptance tests
- Invalid input rejection tests
- Edge case tests (boundary values)
- Serialization/deserialization round-trip tests

### 2C. Generator (13 files — determinism is key)

| Component | Deterministic? | Test Approach |
|-----------|---------------|---------------|
| `DeterministicRNG` | ✅ Seed-based | Same seed → same output |
| `BatchGenerator` | ✅ Uses DeterministicRNG | Same config → same data |
| `DatasetGenerator` | ✅ Uses DeterministicRNG | Full pipeline determinism |
| `ScenarioDefinition` | ✅ Pure data | Schema validation |
| `validation.py` | ✅ Pure functions | Input/output assertions |

---

## 3. Integration Test Strategy

### 3A. Service Chains

| Chain | Components | Test |
|-------|-----------|------|
| Guardrail → Execution | GuardrailEngine + ResolutionExecutionService | Guardrail passes → execution runs |
| Feedback → Reward | FeedbackService + RewardEngine | Feedback recorded → reward calculated |
| Candidate → Score → Select | CandidateGenerator + Scorer + Selector | Generate → score → best selected |
| Evidence → Graph → Guard | EvidenceGraphBuilder + EvidenceGuard | Evidence builds → guard evaluates |
| Learning → Training → Eval | LearningDatasetBuilder + ModelTrainer + Evaluator | Dataset → train → evaluate |
| Policy → Compare → Promote | PolicyStore + PolicyComparator + PromotionGate | Policy compared → promoted/rejected |

### 3B. Components Requiring Mocks

| Component | What to Mock | Why |
|-----------|-------------|-----|
| `MLflowTrackingService` | MLflow API | External service |
| `MLflowModelRegistry` | MLflow API | External service |
| `LLM Providers` | OpenAI/Ollama API | External service |
| `SimilarityService` | EmbeddingService (optionally) | Can use real in-memory |
| `HistoricalCaseStore` | Database session | ORM dependency |

### 3C. Components Requiring Synthetic Fixtures

| Component | Fixture Type | Source |
|-----------|-------------|--------|
| All services | Synthetic exceptions | `DatasetGenerator` with fixed seed |
| Guardrail tests | Financial records with varying risk | Generated via scenarios |
| Evidence tests | Payment/settlement/refund records | Generated via `FinancialDataAdapter` |
| ML tests | Feature vectors + labels | Generated via `FeatureEngineer` |
| Feedback tests | Workflow outcomes | Generated via `OutcomeService` |

---

## 4. Safety Test Strategy

### 4A. Hard Guardrails (P0)

Every guard must have tests proving it blocks unsafe operations:

| Guard | Test Cases |
|-------|-----------|
| `ExposureGuard` | Amount > limit → REJECTED; Amount < limit → PASS |
| `EvidenceGuard` | Missing evidence → REJECTED; Complete evidence → PASS |
| `ConfidenceGate` | Low confidence → HUMAN_REVIEW; High confidence → AUTO |
| `FallbackGuard` | Unknown pattern → ESCALATE; Known pattern → proceed |
| `GuardrailEngine` | Combined: any single failure → blocked |

### 4B. Safety Regression Tests

```
Test that:
1. Guardrail blocking cannot be bypassed by any input combination
2. AUTO decision requires ALL guards to pass
3. High-value transactions always require human review
4. Novel patterns always escalate
5. Conflicting evidence blocks automation
6. Missing evidence blocks automation
```

### 4C. Phase 6 Boundary Tests

```
Verify:
- No code path converts HUMAN_REVIEW to AUTO
- No code path converts UNRESOLVED to AUTO
- No code path bypasses guardrail evaluation
- GuardrailEngine is the ONLY authority for safety decisions
```

---

## 5. E2E Strategy

### 5A. Full Workflow Test

```
Synthetic Data Generation (seed=42)
  → Batch Reconciliation
  → Exception Detection
  → Evidence Collection
  → ML Classification
  → Similar Case Retrieval
  → Candidate Generation
  → Candidate Scoring
  → Guardrail Evaluation
  → Decision (AUTO/HUMAN_REVIEW/UNRESOLVED)
  → Resolution Execution
  → Verification
  → Reward Calculation
  → Feedback Recording
  → Learning Example Generation
```

### 5B. E2E Test Scenarios

| Scenario | Seed | Expected Outcome | Safety Check |
|----------|------|-----------------|-------------|
| Low-risk fee difference | 42 | AUTO resolution | Guardrails pass |
| High-value refund | 43 | HUMAN_REVIEW | Exposure guard blocks |
| Conflicting evidence | 44 | ESCALATE | Evidence guard blocks |
| Unknown exception type | 45 | ESCALATE | Fallback guard blocks |
| Exact match (no action) | 46 | no_action candidate | Deterministic |
| Multiple candidates | 47 | Best scored selected | Scoring deterministic |

### 5C. Negative E2E Tests

```
- Exception with no evidence → UNRESOLVED
- Exception exceeding exposure limit → REJECTED
- Duplicate resolution attempt → IDEMPOTENT response
- Guardrail failure mid-workflow → graceful rollback
```

---

## 6. Mocking Strategy

### 6A. What NOT to Mock

- Pure in-memory services (use real instances)
- Synthetic data generation (use deterministic seeds)
- Financial calculations (use real implementations)
- Guardrail logic (use real guardrails — never weaken for tests)
- Schema validation (use real Pydantic validation)

### 6B. What TO Mock

| Component | Mock Type | Reason |
|-----------|----------|--------|
| MLflow tracking | `unittest.mock.patch` | External API |
| MLflow registry | `unittest.mock.patch` | External API |
| LLM providers | `unittest.mock.patch` | External API / network |
| Database sessions | SQLite in-memory via `db_test_helper` | Avoid PostgreSQL dependency |
| File system | `tmp_path` fixture | Avoid real file I/O in unit tests |

### 6C. Mock Patterns

```python
# MLflow mock pattern
@patch("app.services.mlflow_tracking.mlflow")
def test_experiment_creation(mock_mlflow):
    mock_mlflow.create_experiment.return_value = 1
    service = MLflowTrackingService()
    result = service.create_experiment("test")
    assert result == 1

# Database mock pattern (using db_test_helper)
def test_feedback_recording():
    session = get_test_session()
    service = FeedbackService(session=session)
    result = service.record_feedback(...)
    assert result.feedback_id is not None
```

---

## 7. Database Test Strategy

### 7A. SQLite In-Memory via db_test_helper

The existing `tests/db_test_helper.py` provides:
- `get_test_session()` — new SQLite session
- `create_all_tables()` — creates all tables in-memory
- `reset_database()` — drop and recreate

**Usage**:
```python
from tests.db_test_helper import get_test_session, create_all_tables

@pytest.fixture(autouse=True)
def setup_db():
    create_all_tables()
    yield
    # cleanup handled by in-memory DB
```

### 7B. Components Needing Database Tests

| Component | DB Operations | Test Approach |
|-----------|--------------|---------------|
| `FeedbackService` | `record_feedback()` | SQLite in-memory |
| `OutcomeService` | `record_outcome()` | SQLite in-memory |
| `HistoricalCaseStore` | `store()`, `search()` | SQLite in-memory |
| `SimilarityService` | `index_case()`, `search()` | SQLite in-memory |
| `PersistenceService` | Batch persistence | SQLite in-memory |

### 7C. Database Test Isolation

Each test must be isolated:
- Use `pytest.fixture(autouse=True)` for per-test DB reset
- Never share DB state between tests
- Always clean up after tests

---

## 8. Determinism Requirements

### 8A. Deterministic Components

| Component | Deterministic? | Seed/Mechanism |
|-----------|---------------|----------------|
| `DeterministicRNG` | ✅ | Explicit seed |
| `BatchGenerator` | ✅ | Passes seed to RNG |
| `DatasetGenerator` | ✅ | Uses BatchGenerator |
| `ReconciliationEngine` | ✅ | Pure computation |
| `FinancialDiffService` | ✅ | Arithmetic |
| `GuardrailEngine` | ✅ | Rule-based |
| `RewardEngine` | ✅ | Lookup table |
| `CandidateScorer` | ✅ | Formula-based |
| `ConfidenceGate` | ✅ | Threshold-based |

### 8B. Non-Deterministic Components

| Component | Non-Deterministic? | Why | Test Approach |
|-----------|-------------------|-----|---------------|
| `SimilarityService` | ⚠️ Depends on embeddings | Embedding model may vary | Use mock embeddings or fixed vectors |
| `ML Classifiers` | ⚠️ Training-dependent | Random initialization | Fix random seeds, assert relative ordering |
| LLM Services | ⚠️ Network-dependent | API responses vary | Mock all LLM calls |

### 8C. Determinism Test Pattern

```python
def test_deterministic_reconciliation():
    """Same input must always produce same output."""
    data = generate_test_data(seed=42)
    result1 = reconcile_batch(data)
    result2 = reconcile_batch(data)
    assert result1 == result2
```

---

## 9. Financial Correctness Requirements

### 9A. Amount Calculations

All financial amounts must be tested with:
- Zero amounts
- Positive amounts
- Maximum amounts (₹100,000 / 10,000,000 paise)
- Negative amounts (where applicable)
- Rounding behavior
- Integer overflow protection

### 9B. Guardrail Thresholds

| Threshold | Test |
|-----------|------|
| Exposure limit | Amount > limit → blocked |
| Exposure limit | Amount = limit → blocked |
| Exposure limit | Amount < limit → passes |
| Confidence threshold | Below → HUMAN_REVIEW |
| Confidence threshold | Above → AUTO (if other guards pass) |

### 9C. Financial Accuracy Tests

```
- Expected amount matches actual amount in reconciliation
- Difference calculation is correct (actual - expected)
- Adjustment amounts are within bounds
- Reward values match outcome categories
- Financial impact aggregation is correct
```

---

## 10. Test Data Strategy

### 10A. Synthetic Data Generation

Use the existing `DatasetGenerator` with fixed seeds:

```python
from app.generator.orchestrator import DatasetGenerator

@pytest.fixture
def synthetic_batch():
    gen = DatasetGenerator(seed=42)
    return gen.generate(num_merchants=3, num_cases=10)
```

### 10B. Test Fixtures

| Fixture | Purpose | Source |
|---------|---------|--------|
| `synthetic_batch` | Full batch of financial records | `DatasetGenerator(seed=42)` |
| `single_exception` | One exception with related records | Extract from batch |
| `guardrail_input` | Exception + candidate for guard testing | Constructed |
| `feedback_records` | Multiple feedback records | Generated |
| `ml_features` | Feature vectors for ML tests | `FeatureEngineer` |

### 10C. Edge Case Data

| Edge Case | Test Data |
|-----------|-----------|
| Zero amount | `adjustment_paise = 0` |
| Maximum amount | `adjustment_paise = 10_000_000` |
| Empty evidence | `evidence = []` |
| Conflicting evidence | Two records with opposite amounts |
| Unknown exception type | Type not in training data |
| Duplicate case ID | Same ID submitted twice |

---

## 11. Expected Test Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run only unit tests (fast)
python -m pytest tests/ -v -m "not integration and not e2e"

# Run safety tests specifically
python -m pytest tests/test_guardrail*.py tests/test_exposure_guard.py tests/test_confidence_gate.py tests/test_evidence_guard.py tests/test_fallback_guard.py -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html

# Run a specific test file
python -m pytest tests/test_guardrail_engine.py -v

# Run tests matching a pattern
python -m pytest tests/ -k "guardrail" -v

# Run in parallel (if pytest-xdist installed)
python -m pytest tests/ -n auto
```

---

## 12. Recommended Test Directory Structure

```
tests/
├── conftest.py                          # Shared fixtures
├── db_test_helper.py                    # SQLite in-memory helper
├── fixtures/                            # Shared test data
│   ├── __init__.py
│   ├── synthetic_data.py               # Batch/exception generators
│   ├── guardrail_fixtures.py           # Guard test scenarios
│   └── financial_fixtures.py           # Amount/edge case data
├── unit/                                # Pure unit tests
│   ├── test_schemas.py                 # All Pydantic schema tests
│   ├── test_guardrail_engine.py
│   ├── test_reward_engine.py
│   ├── test_financial_diff.py
│   ├── test_candidate_scorer.py
│   ├── test_confidence_gate.py
│   ├── test_decision_matrix.py
│   ├── test_evidence_graph.py
│   ├── test_evidence_guard.py
│   ├── test_evidence_quality.py
│   ├── test_exposure_guard.py
│   ├── test_fallback_guard.py
│   ├── test_feedback.py
│   ├── test_execution.py
│   ├── test_verification.py
│   ├── test_resolution_verification.py
│   ├── test_learning_metrics.py
│   ├── test_learning_dataset.py
│   ├── test_idempotency.py
│   ├── test_audit_log.py
│   └── test_policy_learning.py
├── integration/                         # Service chain tests
│   ├── test_guardrail_execution.py
│   ├── test_feedback_reward.py
│   ├── test_candidate_pipeline.py
│   ├── test_evidence_guard_chain.py
│   ├── test_learning_pipeline.py
│   └── test_policy_promotion.py
├── e2e/                                 # End-to-end tests
│   ├── test_full_workflow.py
│   ├── test_workflow_safety.py
│   ├── test_workflow_llm_disabled.py
│   └── test_workflow_llm_enabled.py
├── safety/                              # Safety regression tests
│   ├── test_guardrail_bypass.py
│   ├── test_no_auto_override.py
│   ├── test_exposure_limits.py
│   └── test_phase6_boundary.py
└── regression/                          # Regression tests
    └── test_pre_phase14.py             # Verify existing 3562 tests still pass
```

---

## 13. Implementation Order

### Phase 14.1 — Test Infrastructure
1. Create `conftest.py` with shared fixtures
2. Create `tests/fixtures/` with synthetic data generators
3. Configure `pytest.ini` with markers, timeout, parallel settings
4. Verify `db_test_helper.py` works with all model imports

### Phase 14.2 — Unit Tests (Safety-Critical First)
1. `GuardrailEngine` — all guard combinations
2. `ExposureGuard` — amount threshold tests
3. `EvidenceGuard` — missing/conflicting evidence tests
4. `ConfidenceGate` — threshold boundary tests
5. `RewardEngine` — all reward categories
6. `FeedbackService` — CRUD operations
7. `FinancialDiffService` — amount calculations

### Phase 14.3 — Unit Tests (Resolution Pipeline)
1. `CandidateGenerator` — generation logic
2. `CandidateScorer` — scoring formulas
3. `CandidateSelector` — selection logic
4. `DecisionMatrix` — decision combinations
5. `ResolutionEngine` — resolution flow
6. `ExecutionService` — execution state machine
7. `VerificationService` — verification logic

### Phase 14.4 — Unit Tests (Learning Pipeline)
1. `LearningDatasetBuilder` — dataset construction
2. `LearningMetricsService` — metric calculation
3. `BatchLearningLoop` — batch comparison
4. `PolicyLearning` — policy versioning
5. `SelfLearningLoop` — learning cycle

### Phase 14.5 — Integration Tests
1. Guardrail → Execution chain
2. Feedback → Reward → Learning chain
3. Candidate → Score → Select → Guardrail chain
4. Evidence → Graph → Guard chain

### Phase 14.6 — E2E and Safety Tests
1. Full workflow E2E with deterministic seed
2. Safety regression tests
3. Phase 6 boundary tests
4. Negative scenario tests

---

## 14. Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| No pytest.ini configuration | Cannot run markers/parallel | Create pytest.ini first |
| No conftest.py | No shared fixtures | Create conftest.py with common fixtures |
| Database module raises on import if DATABASE_URL unset | Tests may fail on import | Ensure DATABASE_URL is set in test env or mock |
| SimilarityService depends on embeddings | May not be deterministic | Mock embeddings in unit tests |
| 3562 existing tests may have hidden flakiness | Intermittent failures | Run full suite 3x before Phase 14 changes |
| MLflow tests may fail without MLflow server | Test failures | Mock MLflow in all unit tests |
| LLM tests require mock providers | Test failures | Always mock LLM in tests |
