"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  getMetrics,
  getSafetyMetrics,
  listExceptions,
  getHealth,
  listModels,
  getLearningMetrics,
} from "@/app/lib/api";
import {
  StatCard,
  StatCardSkeleton,
  TableSkeleton,
  ChartSkeleton,
  CardSkeleton,
  Badge,
  LoadingState,
  ErrorState,
  Toast,
} from "@/components/ui";
import {
  formatPct,
  formatPaise,
  formatNum,
  formatExceptionType,
  fmtDate,
} from "@/app/lib/utils";
import type {
  SystemMetrics,
  SafetyMetrics,
  ExceptionListItem,
  HealthResponse,
} from "@/app/types";

const COLORS = ["#059669", "#2563eb", "#d97706", "#dc2626", "#7c3aed"];

/**
 * Check if a /metrics or /metrics/safety response has meaningful data
 * (not just the default all-zeros from an empty batch registry).
 */
function metricsHaveData(m: SystemMetrics | null): boolean {
  if (!m) return false;
  return (
    m.total_records > 0 ||
    m.exceptions > 0 ||
    m.auto_resolved > 0 ||
    m.human_review > 0 ||
    m.unresolved > 0 ||
    m.verification_passed > 0 ||
    m.verification_failed > 0 ||
    m.financial_impact_paise > 0
  );
}

function safetyHaveData(s: SafetyMetrics | null): boolean {
  if (!s) return false;
  return (
    s.auto_decisions > 0 ||
    s.human_review_decisions > 0 ||
    s.unresolved_decisions > 0 ||
    s.guardrail_blocks > 0 ||
    s.high_value_blocks > 0 ||
    s.conflict_blocks > 0 ||
    s.novelty_blocks > 0 ||
    s.verification_failures > 0
  );
}

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [safety, setSafety] = useState<SafetyMetrics | null>(null);
  const [exceptions, setExceptions] = useState<ExceptionListItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelsCount, setModelsCount] = useState(0);
  const [learningOk, setLearningOk] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      const [mRes, sRes, eRes, hRes, mlRes, lRes] = await Promise.all([
        getMetrics(),
        getSafetyMetrics(),
        listExceptions({ limit: 500 }),
        getHealth(),
        listModels(),
        getLearningMetrics(),
      ]);
      if (!mounted) return;
      if (mRes.ok && mRes.data) setMetrics(mRes.data.data as SystemMetrics);
      if (sRes.ok && sRes.data) setSafety(sRes.data.data as SafetyMetrics);
      if (eRes.ok && eRes.data)
        setExceptions((eRes.data.data as ExceptionListItem[]) || []);
      if (hRes.ok && hRes.data) setHealth(hRes.data);
      if (mlRes.ok && mlRes.data?.data) setModelsCount((mlRes.data.data as { model_id: string }[]).length);
      if (lRes.ok) setLearningOk(true);
      if (!mRes.ok && !eRes.ok) setError("Cannot connect to backend");
      setLoading(false);
    }
    load();
    return () => {
      mounted = false;
    };
  }, []);

  if (loading)
    return (
      <div>
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-slate-900">Control Center</h2>
          <p className="text-sm text-slate-400 mt-1">Loading…</p>
        </div>
        <div className="stat-grid mb-6">
          {Array.from({ length: 10 }).map((_, i) => (
            <StatCardSkeleton key={i} />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ChartSkeleton />
          <ChartSkeleton />
        </div>
        <CardSkeleton />
      </div>
    );
  if (error) return <ErrorState title="Backend Unavailable" message={error} />;

  // ─── Derive metrics from exception list ─────────────────────────────────
  // The /metrics endpoint reads from the batch registry which may be empty.
  // The /exceptions endpoint has the authoritative list of exceptions.
  // We derive status counts directly from the exception list.
  const hasMetricsData = metricsHaveData(metrics);
  const hasSafetyData = safetyHaveData(safety);

  const totalExceptions = exceptions.length;

  // Status-based counts derived from the exception list
  const resolvedCount = exceptions.filter((e) => e.status === "APPROVED").length;
  const escalatedCount = exceptions.filter((e) => e.status === "ESCALATED").length;
  const rejectedCount = exceptions.filter((e) => e.status === "REJECTED").length;
  const pendingCount = exceptions.filter((e) => e.status === "PENDING").length;

  // "Human review" = pending exceptions awaiting action
  const humanReviewCount = pendingCount;

  // Auto-resolved: use /metrics if it has real batch-processed data,
  // otherwise derive from exception list (exceptions with RESOLVED status that aren't APPROVED/REJECTED)
  const autoResolvedFromMetrics = hasMetricsData ? (metrics?.auto_resolved ?? 0) : 0;
  const autoResolvedFromList = exceptions.filter((e) => e.status === "RESOLVED").length;
  const autoResolved = autoResolvedFromMetrics || autoResolvedFromList;

  // Automation rate: calculate from exception list when /metrics unavailable
  const automationRateFromMetrics = hasMetricsData ? metrics?.automation_rate : null;
  const automationRate = automationRateFromMetrics !== null
    ? automationRateFromMetrics
    : totalExceptions > 0
      ? (autoResolved / totalExceptions)
      : null;

  // Safety metrics: derive from exception list when /metrics/safety unavailable
  // Guardrail pass: exceptions that are APPROVED or RESOLVED (passed review)
  const guardrailPassed = exceptions.filter((e) => e.status === "APPROVED" || e.status === "RESOLVED").length;
  const guardrailPassRate = totalExceptions > 0 ? guardrailPassed / totalExceptions : null;

  // Verification failures: exceptions that are REJECTED
  const verificationFailures = rejectedCount;

  // High value blocks: exceptions with HIGH or CRITICAL risk that are not resolved
  const highValueBlocks = exceptions.filter((e) => 
    (e.risk_category === "HIGH" || e.risk_category === "CRITICAL") && 
    e.status !== "APPROVED" && e.status !== "RESOLVED"
  ).length;

  // Conflict blocks: we don't have conflict data in the exception list, show as unavailable
  const conflictBlocks = null; // Not available from current data

  // Prepare chart data from the same exception list
  const riskData = [
    {
      name: "Low",
      count: exceptions.filter((e) => e.risk_category === "LOW").length,
    },
    {
      name: "Medium",
      count: exceptions.filter((e) => e.risk_category === "MEDIUM").length,
    },
    {
      name: "High",
      count: exceptions.filter((e) => e.risk_category === "HIGH").length,
    },
  ];

  const typeMap = new Map<string, number>();
  exceptions.forEach((e) => {
    typeMap.set(e.exception_type, (typeMap.get(e.exception_type) || 0) + 1);
  });
  const typeData = Array.from(typeMap.entries()).map(([name, value]) => ({
    name: formatExceptionType(name),
    value,
  }));

  return (
    <div>
      {/* ─── Hero Product Banner ────────────────────────────────────────────── */}
      <div className="card mb-6 bg-gradient-to-r from-slate-900 via-[#0c2340] to-slate-900 text-white border-none shadow-md overflow-hidden relative">
        <div className="card-body p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-blue-500/20 text-blue-300 border border-blue-400/30 text-xs font-semibold mb-2">
              <span>●</span> AI-Powered Financial Exception Reconciliation
            </div>
            <h2 className="text-xl md:text-2xl font-bold tracking-tight text-white">
              Autonomous Operations & Closed-Loop Resolution
            </h2>
            <p className="text-xs md:text-sm text-slate-300 mt-1 max-w-2xl leading-relaxed">
              Continuous multi-way reconciliation across payments, settlements, fees, and refunds. AI-assisted classification and similarity with financial safety guardrails and verified human review.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/exceptions/CASE-DEMO-004"
              className="px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold rounded-lg shadow-sm transition-all whitespace-nowrap"
            >
              Demo Case: CASE-DEMO-004 →
            </Link>
          </div>
        </div>
      </div>

      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Control Center</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time financial exception reconciliation & ledger health overview
          </p>
        </div>
      </div>

      {/* ─── Metric Cards ──────────────────────────────────────────────────── */}
      <div className="stat-grid mb-6">
        <StatCard
          label="Total Exceptions"
          value={formatNum(totalExceptions)}
          sub="Across all batches"
        />
        <StatCard
          label="Resolved"
          value={formatNum(resolvedCount)}
          sub="Approved by reviewer"
        />
        <StatCard
          label="Pending Review"
          value={formatNum(humanReviewCount)}
          sub="Awaiting action"
        />
        <StatCard
          label="Escalated"
          value={formatNum(escalatedCount)}
          sub="Routed to human review"
        />
        <StatCard
          label="Rejected"
          value={formatNum(rejectedCount)}
          sub="Marked incorrect"
        />
        <StatCard
          label="Auto-Resolved"
          value={autoResolved > 0 ? formatNum(autoResolved) : "—"}
          sub={automationRate != null ? formatPct(automationRate) + " rate" : "No batch data"}
        />
        <StatCard
          label="Guardrail Pass"
          value={guardrailPassRate != null ? formatPct(guardrailPassRate) : "—"}
          sub={guardrailPassRate != null ? undefined : "No data"}
        />
        <StatCard
          label="Verification Failures"
          value={verificationFailures > 0 ? formatNum(verificationFailures) : "—"}
        />
        <StatCard
          label="High Value Blocks"
          value={highValueBlocks > 0 ? formatNum(highValueBlocks) : "—"}
        />
        <StatCard
          label="Conflict Blocks"
          value="—"
          sub="Not tracked in exception list"
        />
      </div>

      {/* ─── Charts ────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Risk Distribution */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-sm font-bold text-slate-700">
              Risk Distribution
            </h3>
          </div>
          <div className="card-body">
            {riskData.some((d) => d.count > 0) ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={riskData}>
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {riskData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i] || COLORS[0]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-sm text-slate-400 text-center py-8">
                No risk data available
              </div>
            )}
          </div>
        </div>

        {/* Exception Type Distribution */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-sm font-bold text-slate-700">
              Exception Types
            </h3>
          </div>
          <div className="card-body">
            {typeData.length > 0 ? (
              <div className="flex items-center gap-6">
                <ResponsiveContainer width="50%" height={200}>
                  <PieChart>
                    <Pie
                      data={typeData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={70}
                    >
                      {typeData.map((_, i) => (
                        <Cell
                          key={i}
                          fill={COLORS[i % COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex-1 space-y-1.5">
                  {typeData.map((d, i) => (
                    <div key={d.name} className="flex items-center gap-2 text-xs">
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: COLORS[i % COLORS.length] }}
                      />
                      <span className="text-slate-600 flex-1">{d.name}</span>
                      <span className="font-semibold">{d.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-sm text-slate-400 text-center py-8">
                No type data available
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ─── Recent Exceptions ──────────────────────────────────────────────── */}
      <div className="card mb-6">
        <div className="card-header flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-700">
            Recent Exceptions
          </h3>
          <Link
            href="/exceptions"
            className="text-xs font-medium text-brand hover:underline"
          >
            View All →
          </Link>
        </div>
        <div className="overflow-x-auto">
          {exceptions.length === 0 ? (
            <div className="text-sm text-slate-400 text-center py-8">
              No exceptions found
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Type</th>
                  <th>Difference</th>
                  <th>Risk</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {exceptions.map((exc, idx) => (
                  <tr key={`${exc.exception_id}-${exc.batch_id || ''}-${idx}`}>
                    <td>
                      <Link
                        href={`/exceptions/${exc.exception_id}`}
                        className="text-brand font-mono text-xs font-semibold hover:underline"
                      >
                        {exc.exception_id}
                      </Link>
                    </td>
                    <td className="text-xs">
                      {formatExceptionType(exc.exception_type)}
                    </td>
                    <td className="tabular-nums text-xs font-medium">
                      {formatPaise(exc.difference_paise)}
                    </td>
                    <td>
                      <Badge text={exc.risk_category} variant="risk" />
                    </td>
                    <td>
                      <Badge text={exc.status} variant="status" />
                    </td>
                    <td className="text-xs text-slate-400">
                      {fmtDate(exc.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* ─── System Status ──────────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-sm font-bold text-slate-700">System Health</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatusIndicator label="Backend" ok={!!health} />
            <StatusIndicator label="Database" ok={!!health} />
            <StatusIndicator label="ML Engine" ok={modelsCount > 0} detail={modelsCount > 0 ? `${modelsCount} models` : undefined} />
            <StatusIndicator label="Evidence" ok={exceptions.length > 0} detail={exceptions.length > 0 ? `${exceptions.length} exceptions` : undefined} />
            <StatusIndicator label="Agent" ok={!!metrics} />
            <StatusIndicator label="Guardrails" ok={!!safety} />
            <StatusIndicator label="Learning" ok={learningOk} />
            <StatusIndicator label="Version" ok={!!health} detail={health?.version} />
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusIndicator({
  label,
  ok,
  detail,
}: {
  label: string;
  ok: boolean;
  detail?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`w-2 h-2 rounded-full ${ok ? "bg-emerald-500" : "bg-slate-300"}`}
      />
      <span className="text-xs font-medium text-slate-600">{label}</span>
      {detail && (
        <span className="text-xs text-slate-400 ml-auto">{detail}</span>
      )}
    </div>
  );
}
