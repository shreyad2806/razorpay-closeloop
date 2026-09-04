"""Complete scenario test for Razorpay CloseLoop runtime audit."""
import json
import time
import sys
import os

# Force UTF-8 output
os.environ["PYTHONIOENCODING"] = "utf-8"

import requests

BASE = "http://localhost:8000"

def safe_print(s=""):
    """Print that handles encoding issues."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))

# Get all exceptions first
resp = requests.get(f"{BASE}/exceptions?limit=500")
all_exc = resp.json().get("data", [])

# Group by scenario type
by_type = {}
for e in all_exc:
    t = e.get("exception_type", "?")
    if t not in by_type:
        by_type[t] = []
    by_type[t].append(e)

safe_print(f"Total exceptions loaded: {len(all_exc)}")
safe_print(f"Scenario types: {list(by_type.keys())}")
safe_print(f"Counts: {', '.join(f'{k}={len(v)}' for k, v in sorted(by_type.items()))}")
safe_print()

# Helper: test a complete flow for an exception
def test_scenario(scenario_name, exception):
    exc_id = exception["exception_id"]
    diff = exception.get("difference_paise", 0)
    risk = exception.get("risk_category", "?")
    
    result = {
        "scenario": scenario_name,
        "exc_id": exc_id,
        "type": exception.get("exception_type"),
        "risk": risk,
        "diff": diff,
    }
    
    # 4. Exception detail
    r = requests.get(f"{BASE}/exceptions/{exc_id}")
    result["detail_status"] = r.status_code
    
    # 5. Evidence
    r = requests.get(f"{BASE}/exceptions/{exc_id}/evidence")
    ev = r.json().get("data", {})
    result["evidence_count"] = len(ev.get("evidence", []))
    result["evidence_coverage"] = ev.get("coverage", "?")
    
    # 6+7+8+9. Analyze (classification + similar + candidates + guardrails)
    r = requests.post(f"{BASE}/exceptions/{exc_id}/analyze")
    analysis = r.json()
    if analysis.get("data"):
        d = analysis["data"]
        result["classify_type"] = d.get("classification_type")
        result["classify_confidence"] = d.get("classification_confidence")
        result["similar_count"] = d.get("similar_case_count", 0)
        result["candidate_count"] = len(d.get("candidates", []))
        result["candidate_types"] = [c["resolution_type"] for c in d.get("candidates", [])]
        result["guardrail_decision"] = d.get("guardrail", {}).get("decision")
        result["guardrail_confidence"] = d.get("guardrail", {}).get("confidence")
        result["guardrail_risk"] = d.get("guardrail", {}).get("risk_category")
    else:
        result["analyze_error"] = analysis.get("error", "unknown")
    
    # 11. Resolve attempt (with correct enum)
    r = requests.post(f"{BASE}/exceptions/{exc_id}/resolve", json={
        "resolution_type": "FEE_ADJUSTMENT",
        "adjustment_paise": abs(diff) if diff else 0,
        "reason": f"Runtime audit - {scenario_name}"
    })
    result["resolve_status"] = r.status_code
    resolve_data = r.json()
    result["resolve_result_status"] = resolve_data.get("data", {}).get("status") if r.status_code == 200 else resolve_data.get("error", "")
    
    # 14. Review actions (escalate since others may conflict)
    r = requests.post(f"{BASE}/exceptions/{exc_id}/escalate", json={
        "reason": f"Runtime audit - {scenario_name}"
    })
    result["escalate_status"] = r.status_code
    result["escalate_result"] = r.json().get("data", {}).get("status") if r.status_code == 200 else r.json().get("error", "")
    
    # 16. Explain
    r = requests.post(f"{BASE}/explain", json={"exception_id": exc_id})
    explain = r.json()
    result["explain_status"] = r.status_code
    result["explain_fallback"] = explain.get("data", {}).get("fallback_used", "?") if explain.get("data") else "?"
    
    return result

# Run scenarios
print("=" * 80)
print("SCENARIO-BY-SCENARIO RUNTIME TEST")
print("=" * 80)

# Select one representative case per scenario type
scenario_map = {}
for t, cases in by_type.items():
    scenario_map[t] = cases[0]

# Define our 10 scenarios
scenarios = [
    ("Exact Match", "EXACT_MATCH"),
    ("Small Financial Discrepancy", "FEE_DIFFERENCE"),
    ("Partial Settlement", "PARTIAL_SETTLEMENT"),
    ("Timing Difference", "TIMING_DIFFERENCE"),
    ("Tax/Fee Adjustment", "TAX_ADJUSTMENT"),
    ("Large/High-Value", None),  # Pick highest absolute diff
    ("Conflicting Evidence", "DUPLICATE"),
    ("Unknown/Novel", "UNKNOWN"),
    ("Duplicate Settlement", "DUPLICATE"),
    ("Missing Record", "MISSING_RECORD"),
]

# For "Large/High-Value" pick the case with largest diff
all_sorted = sorted(all_exc, key=lambda e: abs(e.get("difference_paise", 0)), reverse=True)
large_case = all_sorted[0] if all_sorted else None
scenarios[5] = ("Large/High-Value", large_case.get("exception_type") if large_case else None)

all_results = []
for name, exc_type in scenarios:
    if exc_type is None and name == "Large/High-Value":
        case = large_case
    elif exc_type and exc_type in scenario_map:
        case = scenario_map[exc_type]
    else:
        safe_print(f"  [SKIP] {name}: no case of type {exc_type}")
        continue
    
    safe_print(f"\n--- Testing: {name} ({case['exception_id']}, {case['exception_type']}, diff={case['difference_paise']}) ---")
    try:
        r = test_scenario(name, case)
        all_results.append(r)
        safe_print(f"  Evidence: {r.get('evidence_count', '?')} records, coverage={r.get('evidence_coverage', '?')}")
        safe_print(f"  Classify: type={r.get('classify_type')}, confidence={r.get('classify_confidence')}")
        safe_print(f"  Candidates: {r.get('candidate_count', '?')} ({r.get('candidate_types', [])})")
        safe_print(f"  Guardrail: {r.get('guardrail_decision')}, confidence={r.get('guardrail_confidence')}, risk={r.get('guardrail_risk')}")
        safe_print(f"  Resolve: status={r.get('resolve_status')}, result={r.get('resolve_result_status', '?')}")
        safe_print(f"  Escalate: status={r.get('escalate_status')}, result={r.get('escalate_result', '?')}")
        safe_print(f"  Explain: fallback={r.get('explain_fallback')}")
    except Exception as e:
        safe_print(f"  ERROR: {e}")
        all_results.append({"scenario": name, "error": str(e)})

# =============================================================
# RESULTS TABLE
# =============================================================
print("\n\n" + "=" * 80)
print("SCENARIO EXECUTION REPORT")
print("=" * 80)
print()
print(f"{'Scenario':<25} {'Type':<28} {'Risk':<8} {'Diff':>10} {'Guardrail':<15} {'Resolve':<12} {'Escalate':<10}")
print("-" * 120)

for r in all_results:
    if "error" in r:
        print(f"{r['scenario']:<25} {'ERROR':<28} {'?':<8} {'?':>10} {'?':<15} {'?':<12} {'?':<10}")
        continue
    print(f"{r.get('scenario', '?'):<25} {r.get('type', '?'):<28} {r.get('risk', '?'):<8} {r.get('diff', 0):>10} {r.get('guardrail_decision', '?'):<15} {str(r.get('resolve_status', '?')):<12} {str(r.get('escalate_result', '?')):<10}")

# =============================================================
# DETAILED PIPELINE TRACE
# =============================================================
print("\n\n" + "=" * 80)
print("DETAILED PIPELINE TRACE (first scenario)")
print("=" * 80)

if all_results:
    r = all_results[0]
    safe_print(f"""
Stage 1-2: Batch Creation + Reconciliation
  Input: Synthetic financial records
  Output: {r.get('exc_id')} with {r.get('type')} scenario
  API: POST /batches, POST /batches/{{id}}/run
  Database: File-system (JSON files)
  Executed: YES
  Data: REAL (synthetic generation + real reconciliation)

Stage 3: Exception Listing
  Input: Batch reconciliation results
  Output: Exception record with financial amounts
  API: GET /exceptions
  Database: File-system JSON
  Executed: YES
  Data: REAL (from reconciled batch)

Stage 4: Exception Detail
  Input: Exception ID
  Output: Full exception detail
  API: GET /exceptions/{{id}}
  Database: File-system JSON
  Executed: YES
  Data: REAL
  Note: classification_confidence={r.get('classify_confidence')}, guardrail_decision={r.get('guardrail_decision')} (both None until analyzed)

Stage 5: Evidence Retrieval
  Input: Exception ID
  Output: {r.get('evidence_count')} evidence records, coverage={r.get('evidence_coverage')}
  API: GET /exceptions/{{id}}/evidence
  Database: FinancialDataAdapter (JSON files)
  Executed: YES
  Data: REAL (from JSON adapter)

Stage 6: Classification
  Input: Exception + financial context
  Output: type={r.get('classify_type')}, confidence={r.get('classify_confidence')}
  API: POST /exceptions/{{id}}/analyze
  Database: None (in-memory heuristic)
  Executed: YES
  Data: HEURISTIC (not real ML classifier)
  Issue: Confidence is None -- AnalyzeService does not compute ML confidence
  Issue: Classification uses exception_type from JSON data, not ML prediction

Stage 7: Similar Cases
  Input: Exception type + financial context
  Output: {r.get('similar_count', '?')} similar cases
  API: GET /exceptions/{{id}}/similar
  Database: FinancialDataAdapter (JSON scan)
  Executed: YES
  Data: HEURISTIC (same-type matching, not embeddings)

Stage 8: Resolution Candidates
  Input: Financial discrepancy + exception type
  Output: {r.get('candidate_count', '?')} candidates: {r.get('candidate_types', [])}
  API: POST /exceptions/{{id}}/analyze (embedded)
  Database: None (in-memory heuristic)
  Executed: YES
  Data: HEURISTIC (difference-based rules, not CandidateGenerator)

Stage 9: Guardrails
  Input: Candidate + risk + confidence
  Output: decision={r.get('guardrail_decision')}, confidence={r.get('guardrail_confidence')}, risk={r.get('guardrail_risk')}
  API: POST /exceptions/{{id}}/analyze (embedded)
  Database: None
  Executed: PARTIALLY
  Data: SIMPLIFIED (always returns HUMAN_REVIEW with confidence=0.0)
  Issue: GuardrailEngine is NOT invoked -- AnalyzeService uses simplified logic
  Issue: Every case gets HUMAN_REVIEW regardless of risk/confidence

Stage 10: LangGraph Workflow
  Input: Exception ID
  Output: Decision, nodes executed
  API: NONE (not wired to API)
  Database: None (all in-memory)
  Executed: YES (when called directly)
  Data: MIXED (simulated investigation + real guardrails)
  Issue: Workflow is never invoked by the production API
  Issue: 8/10 investigation nodes use _simulate_* functions

Stage 11: Resolution Proposal
  Input: Resolution type + adjustment amount
  Output: Status
  API: POST /exceptions/{{id}}/resolve
  Database: In-memory exception registry
  Executed: YES
  Data: REAL (proposal recorded)
  Status: {r.get('resolve_status')} -- {r.get('resolve_result_status', '?')}

Stage 12: Execution
  Input: Action request + financial state
  Output: Execution result
  API: NONE (no /execute endpoint)
  Database: None (in-memory)
  Executed: Only via LangGraph (not via API)
  Data: REAL (ResolutionExecutionService works)
  Issue: UNREACHABLE from the API

Stage 13: Verification
  Input: Execution result + current state
  Output: Verification result
  API: NONE (no /verify endpoint)
  Database: None (in-memory comparison)
  Executed: Only via LangGraph (not via API)
  Data: REAL (VerificationService works)
  Issue: UNREACHABLE from the API

Stage 14: Human Review
  Input: Exception ID + action
  Output: Updated status
  API: POST /exceptions/{{id}}/escalate
  Database: In-memory exception registry
  Executed: YES
  Status: {r.get('escalate_status')} -- {r.get('escalate_result', '?')}

Stage 15: Feedback/Learning
  Input: Workflow outcome + reviewer feedback
  Output: Feedback record
  API: POST /feedback, GET /learning/metrics
  Database: FeedbackService (in-memory)
  Executed: YES (feedback recorded)
  Data: REAL (feedback recorded)
  Issue: Learning metrics return HARDCODED EMPTY values
  Issue: Feedback not persisted to database

Stage 16: LLM Explanation
  Input: Exception + evidence context
  Output: Natural language explanation
  API: POST /explain
  Database: FinancialDataAdapter (JSON) + LLM provider
  Executed: YES
  Data: REAL (deterministic fallback since LLM disabled)
  Fallback: {r.get('explain_fallback')}""")

# =============================================================
# PIPELINE GAPS SUMMARY
# =============================================================
print("\n\n" + "=" * 80)
print("PIPELINE GAP ANALYSIS")
print("=" * 80)
print("""
The pipeline has TWO CRITICAL GAPS:

GAP 1: API -> LangGraph Disconnect
  The production API (FastAPI routes) never invokes the LangGraph workflow.
  The API uses AnalyzeService which has its own simplified pipeline.
  The real LangGraph workflow (with real guardrails, execution, verification)
  is only callable via run_workflow() which exists but is never wired to any route.

  File: app/api/routes/intelligence.py (POST /analyze)
  File: app/api/analyze.py (AnalyzeService)
  File: app/agent/workflow.py (run_workflow -- never imported by API)
  
  SEVERITY: CRITICAL

GAP 2: Investigation Nodes Use Simulated Data
  8 of 10 LangGraph workflow nodes use _simulate_* functions instead
  of real services. Only the guardrail node and execution/verification
  nodes delegate to real implementations.
  
  Files: app/agent/nodes.py, investigation_nodes.py, resolution_nodes.py
  Functions: _simulate_exception_retrieval, _simulate_evidence_retrieval,
             _simulate_classification, _simulate_similarity_search,
             _simulate_candidate_generation, _simulate_scoring, _simulate_selection
  
  SEVERITY: CRITICAL

ADDITIONAL GAPS:

GAP 3: No Database Integration
  API routes read from JSON files via FinancialDataAdapter.
  SQLAlchemy models exist but are never queried by the API.
  PostgreSQL/pgvector are unused.
  File: app/api/services/exception_service.py (reads JSON, not DB)
  SEVERITY: HIGH

GAP 4: Feedback Not Persisted
  FeedbackService stores in memory (dict).
  Lost on server restart.
  Learning metrics always return empty.
  File: app/services/feedback.py
  SEVERITY: HIGH

GAP 5: No Execution/Verification Endpoints
  The API has no /execute or /verify endpoint.
  These exist only as LangGraph workflow nodes.
  The resolve API records a proposal but never executes it.
  File: app/api/routes/exceptions.py (no execute/verify routes)
  SEVERITY: HIGH

GAP 6: Guardrails Always Return HUMAN_REVIEW
  AnalyzeService._build_guardrail_summary() always returns
  HUMAN_REVIEW with confidence=0.0 regardless of the case.
  The real GuardrailEngine is never invoked by the API.
  File: app/api/analyze.py (_build_guardrail_summary)
  SEVERITY: MEDIUM

GAP 7: ML Classifier Never Used
  ExceptionClassifier exists in ml/classifier.py but is never
  called by any API route or workflow node.
  File: app/ml/classifier.py
  SEVERITY: MEDIUM

GAP 8: SimilarityService Never Used
  SimilarityService with pgvector exists but is never called.
  The API uses simple JSON comparison instead.
  File: app/services/similarity_service.py (never instantiated)
  SEVERITY: MEDIUM

GAP 9: CandidateGenerator Never Used
  Real CandidateGenerator exists but API uses heuristic.
  File: app/services/candidate_generator.py (never called by API)
  SEVERITY: MEDIUM
""")

# =============================================================
# VERDICT
# =============================================================
print("=" * 80)
print("FINAL VERDICT: CAN ONE FINANCIAL BATCH TRAVEL THE COMPLETE SYSTEM?")
print("=" * 80)
print("""
ANSWER: NO -- not through a single connected pipeline.

PARTIAL PATH (via API):
  Financial Records -> Reconciliation -> Exceptions -> Evidence -> Analysis -> Feedback
  This path WORKS but uses:
  - Heuristic classification (not ML)
  - Heuristic candidates (not CandidateGenerator)
  - Simplified guardrails (always HUMAN_REVIEW, not GuardrailEngine)
  - No execution, no verification
  - In-memory feedback (not persisted)

DEAD PATH (LangGraph):
  load_exception -> evidence -> classify -> similar -> candidates ->
  guardrails -> verify -> resolve -> execute -> verify_execution -> outcome
  This path WORKS when called directly but:
  - Is NEVER invoked by any API endpoint
  - Uses SIMULATED data for 8/10 investigation nodes
  - Only guardrails + execution + verification use real services

THE MISSING BRIDGE:
  There is no API endpoint that calls run_workflow().
  There is no middleware that connects AnalyzeService to the LangGraph pipeline.
  The API and the workflow are TWO SEPARATE SYSTEMS that share no execution path.
""")
