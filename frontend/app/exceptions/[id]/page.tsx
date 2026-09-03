"use client";

import { useEffect, useState, use } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  getException,
  getEvidence,
  getSimilarCases,
  analyzeException,
  explainException,
  approveException,
  rejectException,
  escalateException,
} from "@/app/lib/api";
import {
  Badge,
  LoadingState,
  ErrorState,
  SectionHeader,
  PipelineProgress,
} from "@/components/ui";
import {
  formatPaise,
  formatPct,
  formatExceptionType,
  fmtDate,
  coverageBadge,
} from "@/app/lib/utils";
import type {
  ExceptionDetail,
  EvidenceResponse,
  SimilarCasesResponse,
  AnalysisResult,
  ExplanationResult,
} from "@/app/types";

type Tab =
  | "summary"
  | "financials"
  | "evidence"
  | "intelligence"
  | "candidates"
  | "guardrails"
  | "similar"
  | "explanation"
  | "review";

const TABS: { key: Tab; label: string; count?: number }[] = [
  { key: "summary", label: "Summary" },
  { key: "financials", label: "Financials" },
  { key: "evidence", label: "Evidence" },
  { key: "intelligence", label: "Intelligence" },
  { key: "candidates", label: "Candidates" },
  { key: "guardrails", label: "Guardrails" },
  { key: "similar", label: "Similar Cases" },
  { key: "explanation", label: "Explanation" },
  { key: "review", label: "Review" },
];

export default function ExceptionDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const searchParams = useSearchParams();
  const batchId = searchParams.get("batch") || undefined;
  const [tab, setTab] = useState<Tab>("summary");
  const [exc, setExc] = useState<ExceptionDetail | null>(null);
  const [evidence, setEvidence] = useState<EvidenceResponse | null>(null);
  const [similar, setSimilar] = useState<SimilarCasesResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Action states
  const [actionLoading, setActionLoading] = useState(false);
  const [actionResult, setActionResult] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      const [eRes, evRes, sRes] = await Promise.all([
        getException(id),
        getEvidence(id),
        getSimilarCases(id),
      ]);
      if (!mounted) return;
      if (eRes.ok && eRes.data?.data) setExc(eRes.data.data as ExceptionDetail);
      else setError(eRes.error || "Exception not found");
      if (evRes.ok && evRes.data?.data) setEvidence(evRes.data.data as EvidenceResponse);
      if (sRes.ok && sRes.data?.data) setSimilar(sRes.data.data as SimilarCasesResponse);
      setLoading(false);
    }
    load();
    return () => { mounted = false; };
  }, [id]);

  // Lazy-load analysis
  async function loadAnalysis() {
    if (analysis) return;
    const { ok, data } = await analyzeException(id);
    if (ok && data?.data) setAnalysis(data.data as AnalysisResult);
  }

  // Lazy-load explanation
  async function loadExplanation() {
    if (explanation) return;
    const { ok, data } = await explainException(id);
    if (ok && data?.data) setExplanation(data.data as ExplanationResult);
  }

  // Tab change handler
  function handleTabChange(t: Tab) {
    setTab(t);
    if (t === "intelligence" || t === "candidates" || t === "guardrails") loadAnalysis();
    if (t === "explanation") loadExplanation();
  }

  // Actions
  async function handleApprove() {
    setActionLoading(true);
    setActionResult(null);
    const { ok } = await approveException(id, {
      approved_by: "reviewer",
      comments: "Approved via dashboard",
    });
    setActionResult(ok ? "Approved successfully" : "Approval failed");
    setActionLoading(false);
    // Refresh exception
    const { data } = await getException(id);
    if (data?.data) setExc(data.data as ExceptionDetail);
  }

  async function handleReject() {
    setActionLoading(true);
    setActionResult(null);
    const { ok } = await rejectException(id, {
      rejected_by: "reviewer",
      reason: "Rejected via dashboard",
    });
    setActionResult(ok ? "Rejected" : "Rejection failed");
    setActionLoading(false);
    const { data } = await getException(id);
    if (data?.data) setExc(data.data as ExceptionDetail);
  }

  async function handleEscalate() {
    setActionLoading(true);
    setActionResult(null);
    const { ok } = await escalateException(id, {
      reason: "Escalated via dashboard for human review",
      escalated_by: "reviewer",
    });
    setActionResult(ok ? "Escalated" : "Escalation failed");
    setActionLoading(false);
    const { data } = await getException(id);
    if (data?.data) setExc(data.data as ExceptionDetail);
  }

  if (loading) return <LoadingState message="Loading exception…" />;
  if (error || !exc)
    return <ErrorState title="Exception Not Found" message={error || id} />;

  const evidenceCount = evidence?.record_count || evidence?.evidence?.length || 0;
  const similarCount = similar?.count || similar?.similar_cases?.length || 0;
  const candidateCount = analysis?.candidates?.length || 0;

  // Update tab counts
  const tabsWithCounts = TABS.map((t) => {
    if (t.key === "evidence") return { ...t, count: evidenceCount || undefined };
    if (t.key === "similar") return { ...t, count: similarCount || undefined };
    if (t.key === "candidates") return { ...t, count: candidateCount || undefined };
    return t;
  });

  return (
    <div>
      {/* ─── Header ──────────────────────────────────────────────────────────── */}
      <div className="mb-4">
        <div className="flex items-center gap-2 text-xs text-slate-400 mb-3">
          <Link href="/exceptions" className="hover:text-brand">
            Exceptions
          </Link>
          <span>›</span>
          <span className="text-slate-700">{exc.exception_id}</span>
          {batchId && (
            <>
              <span>›</span>
              <span className="text-slate-500 font-mono">{batchId}</span>
            </>
          )}
        </div>
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-900">
              Exception Investigation
            </h2>
            <div className="flex items-center gap-3 mt-2 text-sm text-slate-500 flex-wrap">
              <span className="font-mono text-xs bg-slate-100 px-2 py-0.5 rounded">
                {exc.exception_id}
              </span>
              {batchId && (
                <span className="font-mono text-xs bg-blue-50 text-blue-600 border border-blue-200 px-2 py-0.5 rounded">
                  Batch: {batchId}
                </span>
              )}
              <span>{formatExceptionType(exc.exception_type)}</span>
              <Badge text={exc.risk_category} variant="risk" />
              <Badge text={exc.status} variant="status" />
            </div>
          </div>
        </div>
      </div>

      {/* ─── Pipeline ────────────────────────────────────────────────────────── */}
      <div className="card mb-4 overflow-hidden">
        <div className="px-4 py-2">
          <PipelineProgress activeStep={7} />
        </div>
      </div>

      {/* ─── Tabs ────────────────────────────────────────────────────────────── */}
      <div className="tab-bar mb-4">
        {tabsWithCounts.map((t) => (
          <button
            key={t.key}
            className={`tab-btn ${tab === t.key ? "active" : ""}`}
            onClick={() => handleTabChange(t.key)}
          >
            {t.label}
            {t.count != null && (
              <span className="ml-1 text-[10px] text-slate-400">
                ({t.count})
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ─── Tab Content ─────────────────────────────────────────────────────── */}
      {tab === "summary" && (
        <SummaryTab exc={exc} explanation={explanation} batchId={batchId} />
      )}
      {tab === "financials" && (
        <FinancialsTab exc={exc} evidence={evidence} />
      )}
      {tab === "evidence" && <EvidenceTab evidence={evidence} />}
      {tab === "intelligence" && <IntelligenceTab analysis={analysis} />}
      {tab === "candidates" && <CandidatesTab analysis={analysis} />}
      {tab === "guardrails" && <GuardrailsTab analysis={analysis} />}
      {tab === "similar" && <SimilarTab similar={similar} />}
      {tab === "explanation" && <ExplanationTab explanation={explanation} />}
      {tab === "review" && (
        <ReviewTab
          exc={exc}
          onApprove={handleApprove}
          onReject={handleReject}
          onEscalate={handleEscalate}
          actionLoading={actionLoading}
          actionResult={actionResult}
        />
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

function SummaryTab({
  exc,
  explanation,
  batchId,
}: {
  exc: ExceptionDetail;
  explanation: ExplanationResult | null;
  batchId?: string;
}) {
  return (
    <div className="space-y-6">
      {/* Overview */}
      <div className="card">
        <div className="card-header">
          <SectionHeader title="Exception Overview" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <InfoItem label="Exception ID" value={exc.exception_id} mono />
            <InfoItem label="Batch ID" value={batchId || exc.batch_id || "—"} mono />
            <InfoItem label="Payment ID" value={exc.payment_id} mono />
            <InfoItem label="Merchant" value={exc.merchant_id} mono />
            <InfoItem label="Status" value={exc.status} badge="status" />
            <InfoItem label="Type" value={formatExceptionType(exc.exception_type)} />
            <InfoItem label="Risk" value={exc.risk_category} badge="risk" />
            <InfoItem
              label="Expected"
              value={formatPaise(exc.expected_amount_paise)}
              highlight
            />
            <InfoItem
              label="Actual"
              value={formatPaise(exc.actual_amount_paise)}
              highlight
            />
            <InfoItem
              label="Difference"
              value={formatPaise(exc.difference_paise)}
              className={exc.difference_paise < 0 ? "text-red-600" : exc.difference_paise > 0 ? "text-emerald-600" : ""}
            />
            <InfoItem
              label="Confidence"
              value={exc.classification_confidence != null ? formatPct(exc.classification_confidence) : "—"}
            />
            <InfoItem label="Guardrail" value={exc.guardrail_decision || "—"} badge={exc.guardrail_decision ? "guardrail" : undefined} />
            <InfoItem label="Created" value={fmtDate(exc.created_at)} />
          </div>
        </div>
      </div>

      {/* AI Explanation Preview */}
      {explanation?.summary && (
        <div className="card">
          <div className="card-header">
            <SectionHeader
              title="AI Explanation"
              subtitle={explanation.fallback_used ? "Template (LLM unavailable)" : undefined}
            />
          </div>
          <div className="card-body">
            <p className="text-sm text-slate-700">{explanation.summary}</p>
            {explanation.uncertainty && (
              <div className="mt-3 text-xs text-amber-600 bg-amber-50 rounded-lg px-3 py-2">
                <strong>⚠ Uncertainty:</strong> {explanation.uncertainty}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function FinancialsTab({
  exc,
  evidence,
}: {
  exc: ExceptionDetail;
  evidence: EvidenceResponse | null;
}) {
  return (
    <div className="space-y-6">
      {/* Discrepancy Summary */}
      <div className="card">
        <div className="card-header">
          <SectionHeader title="Financial Discrepancy" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-3 gap-6 text-center">
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Expected</div>
              <div className="text-xl font-bold text-slate-900">{formatPaise(exc.expected_amount_paise)}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Actual</div>
              <div className="text-xl font-bold text-slate-900">{formatPaise(exc.actual_amount_paise)}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Difference</div>
              <div className={`text-xl font-bold ${exc.difference_paise < 0 ? "text-red-600" : exc.difference_paise > 0 ? "text-emerald-600" : "text-slate-900"}`}>
                {formatPaise(exc.difference_paise)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Evidence Coverage */}
      {evidence && (
        <div className="card">
          <div className="card-header">
            <SectionHeader
              title="Evidence Coverage"
              subtitle={`${evidence.record_count || evidence.evidence.length} records`}
            />
          </div>
          <div className="card-body">
            <div className="flex items-center gap-3 mb-4">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Coverage</span>
              <Badge text={evidence.coverage.replace(/_/g, " ")} className={coverageBadge(evidence.coverage)} />
            </div>
            {evidence.conflicts.length > 0 && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-3">
                <div className="text-xs font-semibold text-red-700 mb-1">Conflicts</div>
                {evidence.conflicts.map((c, i) => (
                  <div key={i} className="text-xs text-red-600">• {c}</div>
                ))}
              </div>
            )}
            {evidence.missing_evidence.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                <div className="text-xs font-semibold text-amber-700 mb-1">Missing Evidence</div>
                {evidence.missing_evidence.map((m, i) => (
                  <div key={i} className="text-xs text-amber-600">• {m}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function EvidenceTab({ evidence }: { evidence: EvidenceResponse | null }) {
  if (!evidence)
    return (
      <div className="card">
        <div className="card-body">
          <div className="text-sm text-slate-400 text-center py-8">
            No evidence data available
          </div>
        </div>
      </div>
    );

  const recordIcons: Record<string, string> = {
    PAYMENT: "💳",
    SETTLEMENT: "🏦",
    REFUND: "↩️",
    FEE: "💰",
    TAX: "🏛️",
    ADJUSTMENT: "⚖️",
  };

  return (
    <div className="space-y-6">
      {/* Evidence Records */}
      <div className="card">
        <div className="card-header">
          <SectionHeader
            title="Financial Evidence"
            subtitle={`${evidence.evidence.length} records · ${formatPaise(evidence.total_amount_paise)} total`}
          />
        </div>
        <div className="card-body">
          <div className="space-y-3">
            {evidence.evidence.map((rec) => (
              <div
                key={rec.record_id}
                className="flex items-center gap-4 p-3 rounded-lg border border-slate-100 hover:border-slate-200 transition-colors"
              >
                <span className="text-xl">
                  {recordIcons[rec.record_type] || "📄"}
                </span>
                <div className="flex-1">
                  <div className="text-sm font-semibold text-slate-800">
                    {rec.record_type} — {rec.record_id}
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">
                    {rec.status || "—"}
                  </div>
                </div>
                <div className="text-sm font-bold tabular-nums">
                  {formatPaise(rec.amount_paise)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Evidence Graph */}
      <div className="card">
        <div className="card-header">
          <SectionHeader title="Evidence Graph" subtitle="Relationship visualization" />
        </div>
        <div className="card-body">
          <div className="flex flex-col items-center gap-1 text-sm">
            <div className="evidence-node font-semibold">Merchant</div>
            <div className="evidence-edge">↓</div>
            {evidence.evidence
              .filter((r) => r.record_type === "PAYMENT")
              .map((r) => (
                <div key={r.record_id} className="flex flex-col items-center">
                  <div className="evidence-node border-blue-200 bg-blue-50">
                    💳 Payment — {r.record_id}
                  </div>
                  <div className="evidence-edge">↓</div>
                </div>
              ))}
            {evidence.evidence
              .filter((r) => r.record_type === "SETTLEMENT")
              .map((r) => (
                <div key={r.record_id} className="flex flex-col items-center">
                  <div className="evidence-node border-emerald-200 bg-emerald-50">
                    🏦 Settlement — {r.record_id}
                  </div>
                  <div className="evidence-edge">↓</div>
                </div>
              ))}
            <div className="flex gap-3 flex-wrap justify-center">
              {evidence.evidence
                .filter((r) => ["REFUND", "FEE", "TAX", "ADJUSTMENT"].includes(r.record_type))
                .map((r) => (
                  <div key={r.record_id} className="evidence-node">
                    {recordIcons[r.record_type]} {r.record_type}
                    <br />
                    <span className="text-xs text-slate-400">{r.record_id}</span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function IntelligenceTab({ analysis }: { analysis: AnalysisResult | null }) {
  if (!analysis)
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading intelligence…" />
        </div>
      </div>
    );

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="card-header">
          <SectionHeader title="ML Classification" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <InfoItem label="Predicted Type" value={formatExceptionType(analysis.classification_type || "UNKNOWN")} />
            <InfoItem label="ML Confidence" value={analysis.classification_confidence != null ? formatPct(analysis.classification_confidence) : "—"} />
            <InfoItem label="Risk" value={analysis.risk || "—"} badge="risk" />
            <InfoItem label="Similar Cases" value={String(analysis.similar_case_count ?? 0)} />
          </div>
        </div>
      </div>
    </div>
  );
}

function CandidatesTab({ analysis }: { analysis: AnalysisResult | null }) {
  if (!analysis)
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading candidates…" />
        </div>
      </div>
    );

  return (
    <div className="card">
      <div className="card-header">
        <SectionHeader
          title="Resolution Candidates"
          subtitle={`${analysis.candidates.length} candidates generated`}
        />
      </div>
      <div className="card-body space-y-3">
        {analysis.candidates.length === 0 ? (
          <div className="text-sm text-slate-400 text-center py-8">
            No candidates generated
          </div>
        ) : (
          analysis.candidates.map((c, i) => (
            <div
              key={i}
              className={`p-4 rounded-lg border ${
                i === 0
                  ? "border-brand/30 bg-brand/5"
                  : "border-slate-200"
              }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-sm font-semibold text-slate-800">
                    {c.resolution_type.replace(/_/g, " ")}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {c.source} · Confidence: {formatPct(c.confidence)}
                  </div>
                  {c.description && (
                    <div className="text-xs text-slate-400 mt-1">
                      {c.description}
                    </div>
                  )}
                </div>
                <div className="text-right">
                  <div className="text-sm font-bold tabular-nums">
                    {formatPaise(c.adjustment_paise)}
                  </div>
                  {i === 0 && (
                    <Badge text="RECOMMENDED" className="mt-1 bg-brand/10 text-brand border-brand/20" />
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function GuardrailsTab({ analysis }: { analysis: AnalysisResult | null }) {
  if (!analysis)
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading guardrails…" />
        </div>
      </div>
    );

  const g = analysis.guardrail;
  const isAuto = g.decision === "AUTO";

  return (
    <div className="space-y-6">
      {/* Decision Badge */}
      <div className="card">
        <div className="card-body text-center py-8">
          <div
            className={`inline-flex items-center gap-2 px-6 py-3 rounded-full text-lg font-bold ${
              isAuto
                ? "bg-emerald-500 text-white"
                : g.decision === "HUMAN_REVIEW"
                  ? "bg-blue-500 text-white"
                  : "bg-red-500 text-white"
            }`}
          >
            {isAuto && "✓ "}
            {g.decision.replace(/_/g, " ")}
          </div>
          <div className="mt-3 text-sm text-slate-500">
            {isAuto
              ? "All safety conditions passed"
              : g.reasons[0] || "Safety conditions not met"}
          </div>
        </div>
      </div>

      {/* Guardrail Details */}
      <div className="card">
        <div className="card-header">
          <SectionHeader title="Guardrail Evaluation" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <InfoItem label="Confidence" value={formatPct(g.confidence)} />
            <InfoItem label="Risk" value={g.risk_category} badge="risk" />
            <InfoItem label="Exposure" value={formatPaise(g.exposure_paise)} />
            <InfoItem label="Decision" value={g.decision} badge="guardrail" />
          </div>
          {g.reasons.length > 0 && (
            <div className="mt-4 space-y-1">
              <div className="text-xs text-slate-400 uppercase tracking-wider mb-2">Reasons</div>
              {g.reasons.map((r, i) => (
                <div key={i} className="text-sm text-slate-600 flex items-start gap-2">
                  <span className="text-slate-300 mt-0.5">•</span>
                  <span>{r}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SimilarTab({ similar }: { similar: SimilarCasesResponse | null }) {
  if (!similar)
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading similar cases…" />
        </div>
      </div>
    );

  return (
    <div className="card">
      <div className="card-header">
        <SectionHeader
          title="Similar Historical Cases"
          subtitle={`${similar.count} candidates · Confidence: ${similar.confidence || "—"}`}
        />
      </div>
      <div className="card-body">
        {similar.similar_cases.length === 0 ? (
          <div className="text-sm text-slate-400 text-center py-8">
            No similar cases found
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Case ID</th>
                  <th>Type</th>
                  <th>Similarity</th>
                  <th>Risk</th>
                  <th>Resolution</th>
                </tr>
              </thead>
              <tbody>
                {similar.similar_cases.map((c) => (
                  <tr key={c.case_id}>
                    <td className="font-mono text-xs font-semibold text-brand">
                      {c.case_id}
                    </td>
                    <td className="text-xs">
                      {formatExceptionType(c.exception_type)}
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-brand rounded-full"
                            style={{ width: `${c.similarity_score * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-medium tabular-nums">
                          {formatPct(c.similarity_score)}
                        </span>
                      </div>
                    </td>
                    <td>
                      {c.risk_category && (
                        <Badge text={c.risk_category} variant="risk" />
                      )}
                    </td>
                    <td className="text-xs">
                      {c.resolution_type
                        ? c.resolution_type.replace(/_/g, " ")
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function ExplanationTab({ explanation }: { explanation: ExplanationResult | null }) {
  if (!explanation)
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading explanation…" />
        </div>
      </div>
    );

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="card-header">
          <SectionHeader
            title="Explanation"
            subtitle={
              explanation.fallback_used
                ? "Template-based (LLM unavailable)"
                : `Generated by ${explanation.llm_model || "LLM"}`
            }
          />
        </div>
        <div className="card-body space-y-4">
          <div>
            <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Summary</div>
            <p className="text-sm text-slate-700">{explanation.summary}</p>
          </div>
          {explanation.reason && (
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Reason</div>
              <p className="text-sm text-slate-700">{explanation.reason}</p>
            </div>
          )}
          {explanation.evidence_summary && (
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Evidence Summary</div>
              <p className="text-sm text-slate-700">{explanation.evidence_summary}</p>
            </div>
          )}
          {explanation.uncertainty && (
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
              <div className="text-xs font-semibold text-amber-700 mb-1">⚠ Uncertainty</div>
              <p className="text-xs text-amber-600">{explanation.uncertainty}</p>
            </div>
          )}
          {explanation.limitations && (
            <div className="text-xs text-slate-400 italic">{explanation.limitations}</div>
          )}
        </div>
      </div>
    </div>
  );
}

function ReviewTab({
  exc,
  onApprove,
  onReject,
  onEscalate,
  actionLoading,
  actionResult,
}: {
  exc: ExceptionDetail;
  onApprove: () => void;
  onReject: () => void;
  onEscalate: () => void;
  actionLoading: boolean;
  actionResult: string | null;
}) {
  return (
    <div className="space-y-6">
      {actionResult && (
        <div
          className={`rounded-lg p-3 text-sm font-medium ${
            actionResult.includes("failed") || actionResult.includes("not")
              ? "bg-red-50 text-red-700 border border-red-200"
              : "bg-emerald-50 text-emerald-700 border border-emerald-200"
          }`}
        >
          {actionResult}
        </div>
      )}

      {/* Approve */}
      <div className="card">
        <div className="card-body">
          <div className="flex items-start gap-4">
            <div className="text-2xl">✅</div>
            <div className="flex-1">
              <h4 className="text-sm font-bold text-slate-800">Approve Resolution</h4>
              <p className="text-xs text-slate-400 mt-0.5 mb-3">
                Confirm the current resolution is correct
              </p>
              <button
                className="btn btn-success"
                onClick={onApprove}
                disabled={actionLoading}
              >
                {actionLoading ? "Processing…" : "Approve"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Reject */}
      <div className="card">
        <div className="card-body">
          <div className="flex items-start gap-4">
            <div className="text-2xl">❌</div>
            <div className="flex-1">
              <h4 className="text-sm font-bold text-slate-800">Reject Resolution</h4>
              <p className="text-xs text-slate-400 mt-0.5 mb-3">
                Mark this resolution as incorrect
              </p>
              <button
                className="btn btn-danger"
                onClick={onReject}
                disabled={actionLoading}
              >
                {actionLoading ? "Processing…" : "Reject"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Escalate */}
      <div className="card">
        <div className="card-body">
          <div className="flex items-start gap-4">
            <div className="text-2xl">⬆️</div>
            <div className="flex-1">
              <h4 className="text-sm font-bold text-slate-800">Escalate</h4>
              <p className="text-xs text-slate-400 mt-0.5 mb-3">
                Route to higher-level human review
              </p>
              <button
                className="btn btn-warning"
                onClick={onEscalate}
                disabled={actionLoading}
              >
                {actionLoading ? "Processing…" : "Escalate"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

function InfoItem({
  label,
  value,
  mono,
  highlight,
  badge,
  className,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
  highlight?: boolean;
  badge?: "risk" | "status" | "guardrail";
  className?: string;
}) {
  return (
    <div>
      <div className="text-[11px] text-slate-400 uppercase tracking-wider mb-1">
        {label}
      </div>
      {badge ? (
        <Badge text={String(value)} variant={badge} />
      ) : (
        <div
          className={`text-sm font-medium ${
            mono ? "font-mono" : ""
          } ${highlight ? "text-slate-900 font-bold" : "text-slate-700"} ${className || ""}`}
        >
          {value || "—"}
        </div>
      )}
    </div>
  );
}
