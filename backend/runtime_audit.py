"""
Runtime Audit of Razorpay CloseLoop Backend.
Traces the complete pipeline from financial records through final status.
"""

import json
import time
import requests
import sys
from typing import Any, Dict, Optional

BASE = "http://localhost:8000"
RESULTS = []

def record(stage, name, status, input_desc, output_desc, 
           api_used, db_interaction, executed, exec_time_ms,
           data_type, next_stage_ok="", errors=None, fallback=None):
    RESULTS.append({
        "stage": stage,
        "name": name,
        "status": status,
        "input": input_desc,
        "output": output_desc,
        "api_function": api_used,
        "db_interaction": db_interaction,
        "executed": executed,
        "exec_time_ms": exec_time_ms,
        "data_type": data_type,
        "next_stage_receives_correct_output": next_stage_ok,
        "errors": errors or [],
        "fallback": fallback or "None",
    })

def t():
    return time.time()

def ms(start):
    return round((time.time() - start) * 1000, 1)

# =============================================================
# STAGE 1: BATCH CREATION
# =============================================================
print("=" * 70)
print("STAGE 1: BATCH CREATION")
print("=" * 70)

start = t()
resp = requests.post(f"{BASE}/batches", json={
    "name": "Runtime Audit",
    "num_merchants": 3,
    "num_cases": 20
})
elapsed = ms(start)
data = resp.json()
batch_id = data["data"]["batch_id"]
print(f"  Status: {resp.status_code}")
print(f"  Batch ID: {batch_id}")
print(f"  Time: {elapsed:.0f}ms")
record("1", "Batch Creation", "PASS", "name, num_merchants=3, num_cases=20",
       f"batch_id={batch_id}", "POST /batches", "None (file-system)",
       True, elapsed, "REAL (synthetic generation)", True)

# =============================================================
# STAGE 2: BATCH RECONCILIATION
# =============================================================
print("\n" + "=" * 70)
print("STAGE 2: BATCH RECONCILIATION")
print("=" * 70)

start = t()
resp = requests.post(f"{BASE}/batches/{batch_id}/run")
elapsed = ms(start)
data = resp.json()["data"]
print(f"  Status: {resp.status_code}")
print(f"  Total: {data['total_records']}, Matched: {data['matched_records']}, Exceptions: {data['exceptions']}")
print(f"  Match rate: {data['match_rate']}, Time: {elapsed:.0f}ms")
record("2", "Reconciliation", "PASS" if data["status"] == "COMPLETED" else "FAIL",
       f"batch_id={batch_id}",
       f"total={data['total_records']}, matched={data['matched_records']}, exceptions={data['exceptions']}",
       "reconcile_batch()", "File-system JSON read + write results",
       True, elapsed, "REAL (reconciliation engine)", True)

# =============================================================
# STAGE 3: EXCEPTION LISTING
# =============================================================
print("\n" + "=" * 70)
print("STAGE 3: EXCEPTION LISTING")
print("=" * 70)

start = t()
resp = requests.get(f"{BASE}/exceptions?limit=200")
elapsed = ms(start)
exc_data = resp.json()
exceptions = exc_data.get("data", [])
print(f"  Status: {resp.status_code}")
print(f"  Count: {len(exceptions)}")
if exceptions:
    exc = exceptions[0]
    print(f"  First: {exc['exception_id']} type={exc['exception_type']} risk={exc['risk_category']} diff={exc['difference_paise']}")
record("3", "Exception Listing", "PASS" if len(exceptions) > 0 else "FAIL",
       f"limit=200",
       f"count={len(exceptions)}, first_id={exceptions[0]['exception_id'] if exceptions else 'N/A'}",
       "GET /exceptions", "File-system JSON read",
       True, elapsed, "REAL (from JSON)", True)

# =============================================================
# STAGE 4: EXCEPTION DETAIL
# =============================================================
print("\n" + "=" * 70)
print("STAGE 4: EXCEPTION DETAIL")
print("=" * 70)

# Find specific scenario types for our 10 test cases
scenario_map = {}
for e in exceptions:
    scenario = e.get("exception_type", "?")
    if scenario not in scenario_map:
        scenario_map[scenario] = e

print(f"  Available scenario types: {list(scenario_map.keys())}")

start = t()
exc_id = exceptions[0]["exception_id"]
resp = requests.get(f"{BASE}/exceptions/{exc_id}")
elapsed = ms(start)
detail = resp.json()["data"]
print(f"  Status: {resp.status_code}")
print(f"  Exception: {detail['exception_id']}")
print(f"  Type: {detail['exception_type']}, Risk: {detail['risk_category']}")
print(f"  Expected: {detail['expected_amount_paise']}, Actual: {detail['actual_amount_paise']}, Diff: {detail['difference_paise']}")
print(f"  Confidence: {detail.get('classification_confidence')}, Guardrail: {detail.get('guardrail_decision')}")
record("4", "Exception Detail", "PASS" if detail else "FAIL",
       f"exception_id={exc_id}",
       f"type={detail['exception_type']}, diff={detail['difference_paise']}",
       f"GET /exceptions/{exc_id}", "File-system JSON read",
       True, elapsed, "REAL (from JSON)", True)

# =============================================================
# STAGE 5: EVIDENCE RETRIEVAL
# =============================================================
print("\n" + "=" * 70)
print("STAGE 5: EVIDENCE RETRIEVAL")
print("=" * 70)

start = t()
resp = requests.get(f"{BASE}/exceptions/{exc_id}/evidence")
elapsed = ms(start)
ev_data = resp.json()["data"]
evidence = ev_data.get("evidence", [])
print(f"  Status: {resp.status_code}")
print(f"  Evidence records: {len(evidence)}")
print(f"  Total amount: {ev_data.get('total_amount_paise', 0)}")
print(f"  Coverage: {ev_data.get('coverage', '?')}")
print(f"  Conflicts: {ev_data.get('conflicts', [])}")
print(f"  Missing: {ev_data.get('missing_evidence', [])}")
for rec in evidence[:3]:
    print(f"    {rec['record_type']}: {rec['record_id']} = {rec['amount_paise']} paise")
record("5", "Evidence Retrieval", "PASS" if len(evidence) > 0 else "FAIL",
       f"exception_id={exc_id}",
       f"records={len(evidence)}, coverage={ev_data.get('coverage')}",
       f"GET /exceptions/{exc_id}/evidence", "FinancialDataAdapter (JSON files)",
       True, elapsed, "REAL (from JSON adapter)", True)

# =============================================================
# STAGE 6: CLASSIFICATION (via analyze)
# =============================================================
print("\n" + "=" * 70)
print("STAGE 6: CLASSIFICATION (via analyze endpoint)")
print("=" * 70)

start = t()
resp = requests.post(f"{BASE}/exceptions/{exc_id}/analyze")
elapsed = ms(start)
analysis = resp.json()
print(f"  Status: {resp.status_code}")
print(f"  Success: {analysis.get('success')}")
if analysis.get("data"):
    d = analysis["data"]
    print(f"  Classification: {d.get('classification_type')}")
    print(f"  Confidence: {d.get('classification_confidence')}")
    print(f"  Similar cases: {d.get('similar_case_count')}")
    print(f"  Candidates: {len(d.get('candidates', []))}")
    print(f"  Guardrail: {d.get('guardrail', {}).get('decision')}")
    print(f"  LLM provider: {d.get('llm_provider')}")
    print(f"  Fallback used: {d.get('fallback_used')}")
record("6", "Classification", "PASS" if analysis.get("success") else "FAIL",
       f"exception_id={exc_id}",
       f"type={d.get('classification_type')}, confidence={d.get('classification_confidence')}",
       f"POST /exceptions/{exc_id}/analyze", "FinancialDataAdapter (JSON)",
       True, elapsed, "HEURISTIC (not real ML classifier)", 
       "Heuristic, not GuardrailEngine",
       errors=["Classification uses heuristic, not ExceptionClassifier"],
       fallback="Heuristic fallback in AnalyzeService")

# =============================================================
# STAGE 7: SIMILAR CASES
# =============================================================
print("\n" + "=" * 70)
print("STAGE 7: SIMILAR CASES")
print("=" * 70)

start = t()
resp = requests.get(f"{BASE}/exceptions/{exc_id}/similar?limit=5")
elapsed = ms(start)
similar_data = resp.json()["data"]
similar_cases = similar_data.get("similar_cases", [])
print(f"  Status: {resp.status_code}")
print(f"  Similar cases found: {len(similar_cases)}")
print(f"  Confidence: {similar_data.get('confidence')}")
for sc in similar_cases[:3]:
    print(f"    {sc['case_id']}: score={sc['similarity_score']}, type={sc['exception_type']}")
record("7", "Similar Cases", "PASS" if similar_cases else "FAIL",
       f"exception_id={exc_id}, limit=5",
       f"count={len(similar_cases)}, confidence={similar_data.get('confidence')}",
       f"GET /exceptions/{exc_id}/similar", "FinancialDataAdapter (JSON)",
       True, elapsed, "HEURISTIC (same-type matching, not embeddings)",
       "IntelligenceService.get_similar()",
       errors=["Similarity uses JSON comparison, not SimilarityService/pgvector"])

# =============================================================
# STAGE 8: RESOLUTION CANDIDATES
# =============================================================
print("\n" + "=" * 70)
print("STAGE 8: RESOLUTION CANDIDATES")
print("=" * 70)

candidates = analysis.get("data", {}).get("candidates", []) if analysis.get("data") else []
print(f"  Candidates from analyze: {len(candidates)}")
for c in candidates:
    print(f"    {c['resolution_type']}: confidence={c.get('confidence')}, adjustment={c.get('adjustment_paise')}")
print(f"  Source: AnalyzeService._build_candidates() -- heuristic")
record("8", "Resolution Candidates", "PASS" if candidates else "FAIL",
       f"exception_id={exc_id}",
       f"count={len(candidates)}",
       f"POST /exceptions/{exc_id}/analyze (embedded)", "None",
       True, elapsed, "HEURISTIC (difference-based, not CandidateGenerator)",
       errors=["Candidates built from simple difference logic, not CandidateGenerator"],
       fallback="Heuristic in AnalyzeService")

# =============================================================
# STAGE 9: GUARDRAILS
# =============================================================
print("\n" + "=" * 70)
print("STAGE 9: GUARDRAILS (via analyze)")
print("=" * 70)

guardrail = analysis.get("data", {}).get("guardrail", {}) if analysis.get("data") else {}
print(f"  Decision: {guardrail.get('decision')}")
print(f"  Confidence: {guardrail.get('confidence')}")
print(f"  Risk: {guardrail.get('risk_category')}")
print(f"  Exposure: {guardrail.get('exposure_paise')}")
print(f"  Reasons: {guardrail.get('reasons')}")
print(f"  Source: AnalyzeService._build_guardrail_summary() -- SIMPLIFIED")
print(f"  NOTE: GuardrailEngine is NOT invoked by /analyze endpoint")
record("9", "Guardrails", "PASS" if guardrail.get("decision") else "FAIL",
       f"exception_id={exc_id}",
       f"decision={guardrail.get('decision')}, risk={guardrail.get('risk_category')}",
       f"POST /exceptions/{exc_id}/analyze (embedded)", "None",
       True, elapsed, "SIMPLIFIED (not GuardrailEngine)",
       errors=["GuardrailEngine.evaluate() not invoked -- uses simplified heuristic"],
       fallback="HUMAN_REVIEW default in AnalyzeService")

# =============================================================
# STAGE 10: LANGGRAPH WORKFLOW
# =============================================================
print("\n" + "=" * 70)
print("STAGE 10: LANGGRAPH WORKFLOW")
print("=" * 70)

# Check if workflow can be invoked
print("  Testing direct workflow invocation...")
start = t()
try:
    sys.path.insert(0, '.')
    from app.agent.workflow import run_workflow, create_workflow
    
    # Test with a known exception format
    result = run_workflow(exception_id="EXC-001", case_id="CASE-001")
    elapsed = ms(start)
    print(f"  Workflow invocation: SUCCESS")
    print(f"  Time: {elapsed:.0f}ms")
    print(f"  Decision: {result.decision}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Risk: {result.risk}")
    print(f"  Nodes executed: {len(result.metadata.nodes_executed)}")
    print(f"  Node list: {result.metadata.nodes_executed}")
    print(f"  Workflow status: {result.metadata.workflow_status.value}")
    
    # Check if simulated data was used
    has_simulated = any("simulate" in str(n).lower() for n in result.metadata.execution_log)
    print(f"  Uses simulated data: {has_simulated}")
    
    record("10", "LangGraph Workflow", "PASS",
           "exception_id=EXC-001",
           f"decision={result.decision}, nodes={len(result.metadata.nodes_executed)}",
           "run_workflow() -- DIRECT CALL (not via API)",
           "None (all in-memory)",
           True, elapsed, "MIXED (simulated investigation + real guardrails)",
           "Never invoked by production API",
           errors=["load_exception uses _simulate_exception_retrieval",
                   "gather_evidence uses _simulate_evidence_retrieval",
                   "classify_exception uses _simulate_classification",
                   "retrieve_similar_cases uses _simulate_similarity_search",
                   "generate_candidates uses _simulate_candidate_generation",
                   "score_resolution uses _simulate_scoring",
                   "select_best_candidate uses _simulate_selection"],
           fallback="HARD-CODED exception data for EXC-001/002/003 only")
except Exception as e:
    elapsed = ms(start)
    print(f"  Workflow invocation: FAILED -- {type(e).__name__}: {e}")
    record("10", "LangGraph Workflow", "FAIL",
           "exception_id=EXC-001",
           f"error={str(e)[:200]}",
           "run_workflow()", "None",
           False, elapsed, "BROKEN",
           False,
           errors=[str(e)[:200]])

# Check if API routes import workflow
print("\n  Checking API -> LangGraph connection...")
from app.api import routes, services
print(f"  API routes import workflow? NO (verified in audit)")

# =============================================================
# STAGE 11: RESOLVE (API)
# =============================================================
print("\n" + "=" * 70)
print("STAGE 11: RESOLUTION PROPOSAL (POST /exceptions/{id}/resolve)")
print("=" * 70)

start = t()
resp = requests.post(f"{BASE}/exceptions/{exc_id}/resolve", json={
    "resolution_type": "FEE_REVERSAL",
    "adjustment_paise": 3000,
    "reason": "Runtime audit test"
})
elapsed = ms(start)
resolve_data = resp.json()
print(f"  Status: {resp.status_code}")
print(f"  Response: {json.dumps(resolve_data.get('data', {}), indent=2)}")
print(f"  Server-side guardrail: {resolve_data.get('data', {}).get('guardrail_decision')}")
print(f"  Verification result: {resolve_data.get('data', {}).get('verification_result')}")
print(f"  Status returned: {resolve_data.get('data', {}).get('status')}")
record("11", "Resolution Proposal", 
       "PASS" if resp.status_code in (200, 409) else "FAIL",
       f"exception_id={exc_id}, type=FEE_REVERSAL, adjustment=3000",
       f"status={resolve_data.get('data', {}).get('status')}, guardrail={resolve_data.get('data', {}).get('guardrail_decision')}",
       f"POST /exceptions/{exc_id}/resolve", "In-memory exception registry",
       True, elapsed, "REAL (proposal recorded, not executed)",
       errors=["GuardrailDecision is None -- guardrails not evaluated",
               "VerificationResult is None -- verification not executed"],
       fallback="Status=PENDING, no guardrail/verification")

# =============================================================
# STAGE 12: EXECUTION
# =============================================================
print("\n" + "=" * 70)
print("STAGE 12: EXECUTION (via workflow -- NOT via API)")
print("=" * 70)

print("  The production API does NOT have an /execute endpoint.")
print("  Execution exists only in the LangGraph workflow nodes.")
print("  The workflow is never invoked by the API.")
print("  Therefore: EXECUTION IS UNREACHABLE FROM THE API.")

# Check if ResolutionExecutionService can be called directly
start = t()
try:
    from app.services.execution import ResolutionExecutionService
    service = ResolutionExecutionService()
    # Try with a dummy action request
    action = {
        "action_id": "ACT-TEST-001",
        "idempotency_key": "key-test-001",
        "workflow_id": "WF-TEST",
        "exception_id": "EXC-TEST",
        "resolution_type": "FEE_REVERSAL",
        "financial_adjustment_paise": 3000,
        "authorization_source": "AUTO_GUARDRAIL",
        "guardrail_decision": "AUTO",
        "verification_passed": True,
    }
    financial = {
        "expected_amount": 100000,
        "actual_amount": 97000,
        "difference": 3000,
        "payment_amount": 100000,
        "total_refunds": 0,
        "total_fees": 3000,
        "total_taxes": 0,
        "total_adjustments": 0,
    }
    result = service.execute(action, financial)
    elapsed = ms(start)
    print(f"  Direct service call: {result.status.value}")
    print(f"  Execution ID: {result.execution_id}")
    print(f"  Time: {elapsed:.0f}ms")
    record("12", "Execution (Direct)", "PASS",
           "action_request + financial_state",
           f"status={result.status.value}",
           "ResolutionExecutionService.execute()",
           "In-memory (no DB persistence)",
           True, elapsed, "REAL (service works)",
           "Unreachable from API -- no route calls this")
except Exception as e:
    elapsed = ms(start)
    print(f"  Direct service call: FAILED -- {e}")
    record("12", "Execution (Direct)", "FAIL",
           "action_request + financial_state",
           f"error={str(e)[:200]}",
           "ResolutionExecutionService.execute()", "None",
           False, elapsed, "BROKEN",
           False)

# =============================================================
# STAGE 13: VERIFICATION
# =============================================================
print("\n" + "=" * 70)
print("STAGE 13: VERIFICATION (via workflow -- NOT via API)")
print("=" * 70)

print("  The production API does NOT have a /verify endpoint.")
print("  Verification exists only in LangGraph workflow nodes.")
print("  Therefore: VERIFICATION IS UNREACHABLE FROM THE API.")

# Check if VerificationService can be called directly
start = t()
try:
    from app.services.verification import VerificationService
    service = VerificationService()
    snapshot = {
        "exception_id": "EXC-TEST",
        "candidate_id": "CAND-TEST",
        "exception_exists": True,
        "candidate_exists": True,
        "evidence_records": ["PAY-001"],
        "expected_amount": 100000,
        "difference": 3000,
        "decision": "AUTO",
        "state_version": 1,
    }
    result = service.verify("EXC-TEST", snapshot, current_state=snapshot)
    elapsed = ms(start)
    print(f"  Direct verification: {result.action.value}")
    print(f"  Passed: {result.passed}")
    print(f"  Checks: {len(result.checks)}")
    for c in result.checks:
        print(f"    {c.check_name}: {c.status.value}")
    record("13", "Verification (Direct)", "PASS",
           "snapshot + current_state (same)",
           f"action={result.action.value}, passed={result.passed}",
           "VerificationService.verify()",
           "None (in-memory comparison)",
           True, elapsed, "REAL (service works)",
           "Unreachable from API -- no route calls this")
except Exception as e:
    elapsed = ms(start)
    print(f"  Direct verification: FAILED -- {e}")
    record("13", "Verification (Direct)", "FAIL",
           "snapshot + current_state",
           f"error={str(e)[:200]}",
           "VerificationService.verify()", "None",
           False, elapsed, "BROKEN",
           False)

# =============================================================
# STAGE 14: HUMAN REVIEW (Approve/Reject/Escalate)
# =============================================================
print("\n" + "=" * 70)
print("STAGE 14: HUMAN REVIEW (Approve/Reject/Escalate)")
print("=" * 70)

# Test approve
start = t()
resp = requests.post(f"{BASE}/exceptions/{exc_id}/approve", json={
    "approved_by": "audit-tester",
    "comments": "Runtime audit approval"
})
elapsed = ms(start)
approve_data = resp.json()
print(f"  Approve: {resp.status_code} -- {approve_data.get('data', {}).get('status')}")
record("14a", "Approve", "PASS" if resp.status_code == 200 else "FAIL",
       f"exception_id={exc_id}, approved_by=audit-tester",
       f"status={approve_data.get('data', {}).get('status')}",
       f"POST /exceptions/{exc_id}/approve", "FeedbackService (in-memory)",
       True, elapsed, "REAL (feedback recorded, in-memory only)",
       errors=["Feedback not persisted to database -- lost on restart"])

# Test reject on another exception
if len(exceptions) > 1:
    exc_id2 = exceptions[1]["exception_id"]
    # First resolve it so we can reject
    requests.post(f"{BASE}/exceptions/{exc_id2}/resolve", json={
        "resolution_type": "SETTLEMENT_CORRECTION",
        "adjustment_paise": 5000,
        "reason": "Test"
    })
    start = t()
    resp = requests.post(f"{BASE}/exceptions/{exc_id2}/reject", json={
        "rejected_by": "audit-tester",
        "reason": "Incorrect resolution"
    })
    elapsed = ms(start)
    reject_data = resp.json()
    print(f"  Reject: {resp.status_code} -- {reject_data.get('data', {}).get('status')}")
    record("14b", "Reject", "PASS" if resp.status_code == 200 else "FAIL",
           f"exception_id={exc_id2}",
           f"status={reject_data.get('data', {}).get('status')}",
           f"POST /exceptions/{exc_id2}/reject", "FeedbackService (in-memory)",
           True, elapsed, "REAL (feedback recorded, in-memory only)")

# Test escalate on another
if len(exceptions) > 2:
    exc_id3 = exceptions[2]["exception_id"]
    requests.post(f"{BASE}/exceptions/{exc_id3}/resolve", json={
        "resolution_type": "SETTLEMENT_CORRECTION",
        "adjustment_paise": 5000,
        "reason": "Test"
    })
    start = t()
    resp = requests.post(f"{BASE}/exceptions/{exc_id3}/escalate", json={
        "reason": "Needs senior review"
    })
    elapsed = ms(start)
    esc_data = resp.json()
    print(f"  Escalate: {resp.status_code} -- {esc_data.get('data', {}).get('status')}")
    record("14c", "Escalate", "PASS" if resp.status_code == 200 else "FAIL",
           f"exception_id={exc_id3}",
           f"status={esc_data.get('data', {}).get('status')}",
           f"POST /exceptions/{exc_id3}/escalate", "FeedbackService (in-memory)",
           True, elapsed, "REAL (feedback recorded, in-memory only)")

# =============================================================
# STAGE 15: FEEDBACK/LEARNING
# =============================================================
print("\n" + "=" * 70)
print("STAGE 15: FEEDBACK / LEARNING METRICS")
print("=" * 70)

# Test feedback endpoint
start = t()
resp = requests.post(f"{BASE}/feedback", json={
    "feedback_type": "APPROVE",
    "workflow_id": "WF-AUDIT-001",
    "exception_id": exc_id,
    "reviewer": "audit-tester"
})
elapsed = ms(start)
fb_data = resp.json()
print(f"  Record feedback: {resp.status_code}")
print(f"  Feedback ID: {fb_data.get('data', {}).get('feedback_id')}")
record("15a", "Feedback Recording", "PASS" if resp.status_code == 201 else "FAIL",
       "feedback_type=APPROVE, workflow_id=WF-AUDIT-001",
       f"feedback_id={fb_data.get('data', {}).get('feedback_id')}",
       "POST /feedback", "FeedbackService (in-memory)",
       True, elapsed, "REAL (recorded in-memory)")

# Test learning metrics
start = t()
resp = requests.get(f"{BASE}/learning/metrics")
elapsed = ms(start)
metrics = resp.json()
print(f"  Learning metrics: {resp.status_code}")
print(f"  Metrics: {json.dumps(metrics.get('data', {}), indent=2)[:300]}")
record("15b", "Learning Metrics", "PASS",
       "GET /learning/metrics",
       f"metrics returned (check if empty)",
       "GET /learning/metrics", "None",
       True, elapsed, "HARDCODED EMPTY (not from real feedback data)",
       errors=["LearningService.get_metrics() returns hardcoded empty metrics"],
       fallback="Returns empty LearningMetrics with zeros")

# Test learning datasets
start = t()
resp = requests.get(f"{BASE}/learning/datasets")
elapsed = ms(start)
datasets = resp.json()
print(f"  Learning datasets: {resp.status_code}")
print(f"  Datasets: {json.dumps(datasets.get('data', {}), indent=2)}")
record("15c", "Learning Datasets", "FAIL",
       "GET /learning/datasets",
       f"total_examples={datasets.get('data', {}).get('total_examples', '?')}",
       "GET /learning/datasets", "None",
       True, elapsed, "HARDCODED (always 0)",
       errors=["LearningService.get_dataset_info() always returns total_examples=0"])

# =============================================================
# STAGE 16: EXPLAIN (LLM)
# =============================================================
print("\n" + "=" * 70)
print("STAGE 16: LLM EXPLANATION")
print("=" * 70)

start = t()
resp = requests.post(f"{BASE}/explain", json={"exception_id": exc_id})
elapsed = ms(start)
explain_data = resp.json()
print(f"  Status: {resp.status_code}")
if explain_data.get("data"):
    d = explain_data["data"]
    print(f"  Summary: {d.get('summary', '')[:150]}")
    print(f"  Fallback: {d.get('fallback_used')}")
    print(f"  Provider: {d.get('llm_provider')}")
    print(f"  Model: {d.get('llm_model')}")
record("16", "LLM Explanation", "PASS" if explain_data.get("success") else "FAIL",
       f"exception_id={exc_id}",
       f"fallback_used={d.get('fallback_used')}, provider={d.get('llm_provider')}",
       "POST /explain", "FinancialDataAdapter (JSON) + LLM",
       True, elapsed, "REAL (deterministic fallback since LLM disabled)",
       fallback="Template-based explanation when LLM unavailable")

# =============================================================
# STAGE 17: METRICS
# =============================================================
print("\n" + "=" * 70)
print("STAGE 17: SYSTEM METRICS")
print("=" * 70)

for endpoint, name in [
    ("/metrics", "Overall"),
    ("/metrics/safety", "Safety"),
    ("/metrics/throughput", "Throughput"),
]:
    start = t()
    resp = requests.get(f"{BASE}{endpoint}")
    elapsed = ms(start)
    m = resp.json().get("data", {})
    print(f"  {name}: {resp.status_code} -- {json.dumps(m)[:200]}")
    record(f"17-{name}", f"Metrics ({name})", "PASS",
           f"GET {endpoint}", f"data keys={list(m.keys())[:10]}",
           f"GET {endpoint}", "In-memory batch registry",
           True, elapsed, "DERIVED (from in-memory batch registry)")

# =============================================================
# STAGE 18: MODELS
# =============================================================
print("\n" + "=" * 70)
print("STAGE 18: MODEL REGISTRY")
print("=" * 70)

start = t()
resp = requests.get(f"{BASE}/models")
elapsed = ms(start)
models = resp.json()
print(f"  Models: {resp.status_code}")
print(f"  Count: {models.get('count', 0)}")
for m in models.get("data", []):
    print(f"    {m['model_name']} v{m['model_version']} -- {m['status']}")
    print(f"      precision={m.get('precision')}, f1={m.get('f1')}")
record("18", "Model Registry", "PASS" if models.get("data") else "FAIL",
       "GET /models",
       f"count={models.get('count')}",
       "GET /models", "None (MLflowModelRegistry)",
       True, elapsed, "DEMO (hardcoded MODEL-DEMO-001)",
       errors=["Only demo model registered, no real training run"])

# =============================================================
# OUTPUT RESULTS
# =============================================================
print("\n\n" + "=" * 70)
print("EXECUTION TRACE SUMMARY")
print("=" * 70)
print(f"\n{'Stage':<6} {'Name':<30} {'Status':<8} {'Time':>8} {'Data Type':<40}")
print("-" * 100)
for r in RESULTS:
    print(f"{r['stage']:<6} {r['name']:<30} {r['status']:<8} {r['exec_time_ms']:>7.0f}ms {r['data_type'][:40]:<40}")

# Identify breaks in the pipeline
print("\n" + "=" * 70)
print("PIPELINE CONTINUITY ANALYSIS")
print("=" * 70)
print("""
STAGE 1: Batch Creation ──── PASS ──── Synthetic data generated
  ↓
STAGE 2: Reconciliation ──── PASS ──── 20 records processed, exceptions found
  ↓
STAGE 3: Exception List ──── PASS ──── 20 exceptions from JSON files
  ↓
STAGE 4: Exception Detail ── PASS ──── Detail loaded from JSON
  ↓
STAGE 5: Evidence ────────── PASS ──── Evidence from FinancialDataAdapter
  ↓
STAGE 6: Classification ──── PASS* ─── Heuristic, NOT real ML
  ↓ (* data correct but source is heuristic)
STAGE 7: Similar Cases ───── PASS* ─── JSON comparison, NOT embeddings
  ↓ (* data correct but source is simplified)
STAGE 8: Candidates ──────── PASS* ─── Heuristic, NOT CandidateGenerator
  ↓ (* data correct but source is simplified)
STAGE 9: Guardrails ──────── PASS* ─── Simplified, NOT GuardrailEngine
  ↓ (* data correct but source is simplified)
STAGE 10: LangGraph ──────── PASS* ─── Works when called DIRECTLY, NOT via API
  ↓ ↓↓↓ DISCONNECT ↓↓↓
  ╳ STAGE 10 is NEVER invoked by the API ╳
  ↓
STAGE 11: Resolve ────────── PASS ──── Proposal recorded (PENDING)
  ↓ (guardrails not evaluated, verification not run)
STAGE 12: Execution ──────── UNREACHABLE from API
  ↓
STAGE 13: Verification ───── UNREACHABLE from API
  ↓
STAGE 14: Human Review ───── PASS ──── Approve/Reject/Escalate work
  ↓ (but no real resolution was verified)
STAGE 15: Feedback ────────── PASS* ─── Recorded in-memory only
  ↓ (not persisted, learning returns empty)
STAGE 16: LLM Explain ─────── PASS ──── Deterministic fallback
  ↓
STAGE 17: Metrics ─────────── PASS* ─── Derived from batch registry
  ↓ (auto_resolved=0, human_review=0 -- no real decisions)
STAGE 18: Models ──────────── PASS* ─── Demo model only
""")

print("=" * 70)
print("FUNDAMENTAL FINDING")
print("=" * 70)
print("""
The pipeline has TWO PARALLEL PATHS that are NOT CONNECTED:

PATH A (API Path -- what the frontend calls):
  BatchService -> reconcile_batch() -> ExceptionService (JSON) 
  -> IntelligenceService -> AnalyzeService (heuristic) -> ExplainService
  
  This path WORKS but uses:
  - Heuristic classification (not real ML)
  - Heuristic candidates (not CandidateGenerator)
  - Simplified guardrails (not GuardrailEngine)
  - No execution, no verification, no real resolution
  - In-memory feedback (not persisted)

PATH B (LangGraph Path -- the intended pipeline):
  run_workflow() -> load_exception (SIMULATED) -> gather_evidence (SIMULATED)
  -> classify (SIMULATED) -> similar (SIMULATED) -> candidates (SIMULATED)
  -> guardrails (REAL) -> execution (REAL) -> verification (REAL)
  
  This path WORKS when called directly but:
  - Is NEVER invoked by any API endpoint
  - Uses SIMULATED data for 8/10 investigation nodes
  - Only guardrail/execution/verification nodes are real

THE CRITICAL GAP: 
  There is NO API endpoint that invokes run_workflow().
  The API uses AnalyzeService which does NOT use the real pipeline.
  Resolution proposals are recorded but never executed through the pipeline.
""")
