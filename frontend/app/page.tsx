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
} from "@/app/lib/api";
import {
  StatCard,
  Badge,
  LoadingState,
  ErrorState,
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

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [safety, setSafety] = useState<SafetyMetrics | null>(null);
  const [exceptions, setExceptions] = useState<ExceptionListItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      const [mRes, sRes, eRes, hRes] = await Promise.all([
        getMetrics(),
        getSafetyMetrics(),
        listExceptions({ limit: 10 }),
        getHealth(),
      ]);
      if (!mounted) return;
      if (mRes.ok && mRes.data) setMetrics(mRes.data.data as SystemMetrics);
      if (sRes.ok && sRes.data) setSafety(sRes.data.data as SafetyMetrics);
      if (eRes.ok && eRes.data)
        setExceptions((eRes.data.data as ExceptionListItem[]) || []);
      if (hRes.ok && hRes.data) setHealth(hRes.data);
      if (!mRes.ok && !eRes.ok) setError("Cannot connect to backend");
      setLoading(false);
    }
    load();
    return () => {
      mounted = false;
    };
  }, []);

  if (loading) return <LoadingState message="Loading Control Center…" />;
  if (error) return <ErrorState title="Backend Unavailable" message={error} />;

  // Prepare chart data
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
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-900">Control Center</h2>
        <p className="text-sm text-slate-400 mt-1">
          Financial exception reconciliation — system overview
        </p>
      </div>

      {/* ─── Metric Cards ──────────────────────────────────────────────────── */}
      <div className="stat-grid mb-6">
        <StatCard
          label="Total Exceptions"
          value={formatNum(metrics?.exceptions ?? exceptions.length)}
          sub="Across all batches"
        />
        <StatCard
          label="Auto-Resolved"
          value={formatNum(metrics?.auto_resolved)}
          sub={formatPct(metrics?.automation_rate) + " automation rate"}
        />
        <StatCard
          label="Human Review"
          value={formatNum(metrics?.human_review)}
          sub="Pending reviewer action"
        />
        <StatCard
          label="Unresolved"
          value={formatNum(metrics?.unresolved)}
          sub="Need investigation"
        />
        <StatCard
          label="Guardrail Pass"
          value={formatPct(safety?.guardrail_pass_rate)}
        />
        <StatCard
          label="Verification Failures"
          value={formatNum(safety?.verification_failures)}
        />
        <StatCard
          label="High Value Blocks"
          value={formatNum(safety?.high_value_blocks)}
        />
        <StatCard
          label="Conflict Blocks"
          value={formatNum(safety?.conflict_blocks)}
        />
        <StatCard
          label="Financial Impact"
          value={formatPaise(metrics?.financial_impact_paise)}
        />
        <StatCard
          label="Match Rate"
          value={formatPct(metrics?.match_rate)}
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
            <StatusIndicator label="ML Engine" ok={true} />
            <StatusIndicator label="Evidence" ok={true} />
            <StatusIndicator label="Agent" ok={true} />
            <StatusIndicator label="LLM" ok={false} />
            <StatusIndicator label="MCP" ok={true} />
            <StatusIndicator
              label="Version"
              ok={true}
              detail={health?.version}
            />
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
