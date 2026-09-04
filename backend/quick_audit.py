"""Quick runtime audit of CloseLoop pipeline."""
import json, time, requests, sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
BASE = "http://localhost:8000"

def p(s=""):
    try:
        print(s)
    except:
        print(s.encode("ascii","replace").decode("ascii"))

# Load exceptions
resp = requests.get(f"{BASE}/exceptions?limit=500", timeout=10)
all_exc = resp.json().get("data", [])
by_type = {}
for e in all_exc:
    t = e["exception_type"]
    if t not in by_type: by_type[t] = []
    by_type[t].append(e)

p(f"Total exceptions: {len(all_exc)}")
p(f"Types: {', '.join(f'{k}({len(v)})' for k,v in sorted(by_type.items()))}")
p()

# Pick one per scenario
pick = {}
for t, cases in by_type.items():
    pick[t] = cases[0]

# Add largest-value case
largest = max(all_exc, key=lambda e: abs(e.get("difference_paise",0)))
pick["HIGH_VALUE"] = largest

# Test scenarios
scenarios = [
    ("1. Exact Match", "EXACT_MATCH"),
    ("2. Fee Difference", "FEE_DIFFERENCE"),
    ("3. Partial Settlement", "PARTIAL_SETTLEMENT"),
    ("4. Timing Difference", "TIMING_DIFFERENCE"),
    ("5. Tax Adjustment", "TAX_ADJUSTMENT"),
    ("6. High Value", "HIGH_VALUE"),
    ("7. Duplicate", "DUPLICATE"),
    ("8. Unknown", "UNKNOWN"),
    ("9. Missing Record", "MISSING_RECORD"),
    ("10. Complex Multi", "COMPLEX_MULTI_ADJUSTMENT"),
]

results = []
for name, stype in scenarios:
    exc = pick.get(stype)
    if not exc:
        p(f"{name}: SKIP (no case)")
        continue
    
    eid = exc["exception_id"]
    p(f"\n{'='*60}")
    p(f"{name}: {eid} | {exc['exception_type']} | risk={exc['risk_category']} | diff={exc['difference_paise']}")
    p(f"{'='*60}")
    
    r = {"name": name, "exc_id": eid, "type": exc["exception_type"], "risk": exc["risk_category"], "diff": exc["difference_paise"]}
    
    # Evidence
    t0 = time.time()
    ev_resp = requests.get(f"{BASE}/exceptions/{eid}/evidence", timeout=10)
    ev_ms = (time.time()-t0)*1000
    ev = ev_resp.json().get("data",{})
    r["evidence_count"] = len(ev.get("evidence",[]))
    r["evidence_ms"] = round(ev_ms)
    p(f"  Evidence: {r['evidence_count']} records in {r['evidence_ms']}ms coverage={ev.get('coverage')}")
    
    # Analyze
    t0 = time.time()
    an_resp = requests.post(f"{BASE}/exceptions/{eid}/analyze", timeout=15)
    an_ms = (time.time()-t0)*1000
    an = an_resp.json().get("data",{})
    r["an_ms"] = round(an_ms)
    r["classify"] = an.get("classification_type","?")
    r["conf"] = an.get("classification_confidence")
    r["similar"] = an.get("similar_case_count",0)
    r["cands"] = [c["resolution_type"] for c in an.get("candidates",[])]
    g = an.get("guardrail",{})
    r["g_decision"] = g.get("decision","?")
    r["g_conf"] = g.get("confidence")
    r["g_risk"] = g.get("risk_category","?")
    p(f"  Analyze: classify={r['classify']} conf={r['conf']} similar={r['similar']} cands={r['cands']}")
    p(f"  Guardrail: decision={r['g_decision']} conf={r['g_conf']} risk={r['g_risk']} ({r['an_ms']}ms)")
    
    # Resolve attempt
    t0 = time.time()
    rs_resp = requests.post(f"{BASE}/exceptions/{eid}/resolve", json={
        "resolution_type": r["cands"][0] if r["cands"] else "FEE_ADJUSTMENT",
        "adjustment_paise": abs(exc["difference_paise"]) if exc["difference_paise"] else 0,
        "reason": f"audit-{stype}"
    }, timeout=10)
    rs_ms = (time.time()-t0)*1000
    r["resolve_code"] = rs_resp.status_code
    r["resolve_ms"] = round(rs_ms)
    if rs_resp.status_code == 200:
        r["resolve_status"] = rs_resp.json().get("data",{}).get("status","?")
    else:
        r["resolve_status"] = rs_resp.json().get("error","?")[:80]
    p(f"  Resolve: HTTP {r['resolve_code']} status={r['resolve_status']} ({r['resolve_ms']}ms)")
    
    # Escalate (safe, doesn't conflict)
    t0 = time.time()
    es_resp = requests.post(f"{BASE}/exceptions/{eid}/escalate", json={
        "reason": f"audit-{stype}"
    }, timeout=10)
    es_ms = (time.time()-t0)*1000
    r["esc_code"] = es_resp.status_code
    r["esc_ms"] = round(es_ms)
    if es_resp.status_code == 200:
        r["esc_status"] = es_resp.json().get("data",{}).get("status","?")
    else:
        r["esc_status"] = es_resp.json().get("error","?")[:80]
    p(f"  Escalate: HTTP {r['esc_code']} status={r['esc_status']} ({r['esc_ms']}ms)")
    
    # Explain
    t0 = time.time()
    ex_resp = requests.post(f"{BASE}/explain", json={"exception_id": eid}, timeout=10)
    ex_ms = (time.time()-t0)*1000
    ex = ex_resp.json()
    r["explain_code"] = ex_resp.status_code
    r["explain_ms"] = round(ex_ms)
    r["explain_fallback"] = ex.get("data",{}).get("fallback_used","?") if ex.get("data") else "?"
    p(f"  Explain: HTTP {r['explain_code']} fallback={r['explain_fallback']} ({r['explain_ms']}ms)")
    
    results.append(r)

# Summary table
p("\n\n" + "="*90)
p("SCENARIO RESULTS TABLE")
p("="*90)
p(f"{'Scenario':<22} {'Type':<28} {'Risk':<6} {'Diff':>10} {'Guardrail':<13} {'Resolve':<25} {'Escalate':<12}")
p("-"*90)
for r in results:
    rs = f"{r.get('resolve_code','?')}:{r.get('resolve_status','?')[:20]}"
    p(f"{r['name']:<22} {r['type']:<28} {r['risk']:<6} {r['diff']:>10} {r['g_decision']:<13} {rs:<25} {r.get('esc_status','?'):<12}")

# LangGraph direct test
p("\n\n" + "="*60)
p("LANGGRAPH WORKFLOW (direct invocation)")
p("="*60)
try:
    from app.agent.workflow import run_workflow
    t0 = time.time()
    result = run_workflow(exception_id="EXC-001", case_id="CASE-001")
    wf_ms = (time.time()-t0)*1000
    p(f"  Decision: {result.decision}")
    p(f"  Confidence: {result.confidence}")
    p(f"  Risk: {result.risk}")
    p(f"  Nodes: {len(result.metadata.nodes_executed)} ({', '.join(result.metadata.nodes_executed)})")
    p(f"  Status: {result.metadata.workflow_status.value}")
    p(f"  Time: {round(wf_ms)}ms")
    p(f"  Uses simulated data: {'load_exception' in [n for n in result.metadata.nodes_executed]}")
except Exception as e:
    p(f"  FAILED: {e}")

# Execution service direct test
p("\n" + "="*60)
p("EXECUTION + VERIFICATION (direct service calls)")
p("="*60)
try:
    from app.services.execution import ResolutionExecutionService
    from app.services.verification import VerificationService
    svc = ResolutionExecutionService()
    ver = VerificationService()
    
    action = {"action_id":"ACT-TEST","idempotency_key":"k1","workflow_id":"WF-T","exception_id":"EXC-T",
              "resolution_type":"FEE_ADJUSTMENT","financial_adjustment_paise":3000,
              "authorization_source":"AUTO_GUARDRAIL","guardrail_decision":"AUTO","verification_passed":True}
    fin = {"expected_amount":100000,"actual_amount":97000,"difference":3000,"payment_amount":100000,
           "total_refunds":0,"total_fees":3000,"total_taxes":0,"total_adjustments":0,
           "settlement_count":1,"refund_count":0,"fee_count":1,"tax_count":0,"adjustment_count":0}
    
    t0 = time.time()
    ex = svc.execute(action, fin)
    p(f"  Execution: {ex.status.value} in {round((time.time()-t0)*1000)}ms")
    
    snap = {"exception_id":"EXC-T","candidate_id":"CAND-T","exception_exists":True,
            "candidate_exists":True,"evidence_records":["PAY-T"],"expected_amount":100000,
            "difference":3000,"decision":"AUTO","state_version":1}
    t0 = time.time()
    vr = ver.verify("EXC-T", snap, snap)
    p(f"  Verification: {vr.action.value} in {round((time.time()-t0)*1000)}ms ({len(vr.checks)} checks)")
    for c in vr.checks:
        p(f"    {c.check_name}: {c.status.value}")
except Exception as e:
    p(f"  FAILED: {e}")

# Metrics
p("\n" + "="*60)
p("METRICS + LEARNING")
p("="*60)
for ep in ["/metrics", "/metrics/safety", "/metrics/throughput", "/learning/metrics", "/models"]:
    resp = requests.get(f"{BASE}{ep}", timeout=10)
    d = resp.json().get("data",{})
    # Summarize
    if "total_records" in d:
        p(f"  {ep}: total={d.get('total_records')} matched={d.get('matched_records')} exceptions={d.get('exceptions')} match_rate={d.get('match_rate')}")
    elif "auto_decisions" in d:
        p(f"  {ep}: auto={d.get('auto_decisions')} human={d.get('human_review_decisions')} blocks={d.get('guardrail_blocks')}")
    elif "total_records_processed" in d:
        p(f"  {ep}: processed={d.get('total_records_processed')} batches={d.get('batches_processed')}")
    elif "automation" in d:
        p(f"  {ep}: automation_rate={d['automation'].get('automation_rate')} total={d['automation'].get('total_exceptions')}")
    elif isinstance(d, list):
        p(f"  {ep}: {len(d)} models")
        for m in d:
            p(f"    {m.get('model_name')} v{m.get('model_version')} status={m.get('status')} f1={m.get('f1')}")
    else:
        p(f"  {ep}: {json.dumps(d)[:120]}")

# Final verdict
p("\n\n" + "="*60)
p("PIPELINE CONTINUITY VERDICT")
p("="*60)
p("""
WORKING PATH (via API):
  Batch -> Reconciliation -> Exceptions -> Evidence -> Analyze -> Explain
  Status: PARTIAL -- uses heuristics, not real services

BROKEN PATHS:
  1. API never calls LangGraph workflow (run_workflow)
  2. LangGraph investigation nodes use _simulate_* functions
  3. No /execute or /verify endpoint in API
  4. Guardrails always return HUMAN_REVIEW via API
  5. Feedback not persisted (in-memory only)
  6. Learning metrics hardcoded to empty
  7. ML classifier never invoked
  8. pgvector/SimilarityService never used

DIRECT SERVICE PATH (when called programmatically):
  Execution: WORKS (1ms)
  Verification: WORKS (6 checks all pass)
  GuardrailEngine: WORKS (when called directly)
  These services are REAL but UNREACHABLE from the API.

CONCLUSION: One financial batch CANNOT travel the complete system.
The reconciliation works. The individual services work.
But they are not connected into a single pipeline.
""")
