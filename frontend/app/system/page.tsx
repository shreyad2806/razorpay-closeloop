"use client";

import { useEffect, useState } from "react";
import {
  getHealth,
  getMetrics,
  getSafetyMetrics,
  getThroughputMetrics,
} from "@/app/lib/api";
import { StatCard, LoadingState, ErrorState, SectionHeader } from "@/components/ui";
import { formatPct, formatNum } from "@/app/lib/utils";
import type {
  HealthResponse,
  SystemMetrics,
  SafetyMetrics,
  ThroughputMetrics,
} from "@/app/types";

const PHASES = [
  { num: 1, name: "Synthetic Financial Data" },
  { num: 2, name: "Deterministic Reconciliation" },
  { num: 3, name: "Financial Evidence" },
  { num: 4, name: "ML Classification + Similarity" },
  { num: 5, name: "Resolution Candidates" },
  { num: 6, name: "Hard Financial Guardrails" },
  { num: 7, name: "LangGraph Agent Workflow" },
  { num: 8, name: "Resolution Execution + Verification" },
  { num: 9, name: "Learning + Feedback" },
  { num: 10, name: "MLflow + Model Registry" },
  { num: 11, name: "MCP Controlled Tools" },
  { num: 12, name: "LLM Explanation Layer" },
  { num: 13, name: "REST API + Documentation" },
  { num: 14, name: "Backend Testing (4,800+ tests)" },
  { num: 15, name: "Production Frontend" },
];

const SYSTEMS = [
  { name: "Backend API", status: true },
  { name: "Database", status: true },
  { name: "ML Engine", status: true },
  { name: "Evidence Layer", status: true },
  { name: "LangGraph Agent", status: true },
  { name: "Guardrails", status: true },
  { name: "Verification", status: true },
  { name: "Learning", status: true },
  { name: "MCP", status: true },
  { name: "LLM", status: false },
];

export default function SystemPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [safety, setSafety] = useState<SafetyMetrics | null>(null);
  const [throughput, setThroughput] = useState<ThroughputMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      const [hRes, mRes, sRes, tRes] = await Promise.all([
        getHealth(),
        getMetrics(),
        getSafetyMetrics(),
        getThroughputMetrics(),
      ]);
      if (!mounted) return;
      if (hRes.ok && hRes.data) setHealth(hRes.data);
      if (mRes.ok && mRes.data?.data) setMetrics(mRes.data.data as SystemMetrics);
      if (sRes.ok && sRes.data?.data) setSafety(sRes.data.data as SafetyMetrics);
      if (tRes.ok && tRes.data?.data) setThroughput(tRes.data.data as ThroughputMetrics);
      if (!hRes.ok) setError("Cannot connect to backend");
      setLoading(false);
    }
    load();
    return () => { mounted = false; };
  }, []);

  if (loading) return <LoadingState message="Loading system info…" />;
  if (error) return <ErrorState title="Backend Unavailable" message={error} />;

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-900">System</h2>
        <p className="text-sm text-slate-400 mt-1">
          System health, safety metrics, and throughput
        </p>
      </div>

      {/* ─── Health + Core Metrics ────────────────────────────────────────────── */}
      <div className="stat-grid mb-6">
        <StatCard
          label="Health Status"
          value={
            <span className={health?.status === "ok" ? "text-emerald-600" : "text-red-600"}>
              {health?.status || "—"}
            </span>
          }
          sub={health?.version}
        />
        <StatCard label="Total Exceptions" value={formatNum(metrics?.exceptions)} />
        <StatCard
          label="Automation Rate"
          value={formatPct(metrics?.automation_rate)}
        />
        <StatCard
          label="Avg Processing"
          value={
            throughput?.avg_processing_time_ms
              ? `${throughput.avg_processing_time_ms.toFixed(0)} ms`
              : "—"
          }
        />
        <StatCard
          label="Records/sec"
          value={throughput?.records_per_second?.toFixed(1) || "—"}
        />
      </div>

      {/* ─── Safety Status ────────────────────────────────────────────────────── */}
      <div className="card mb-6">
        <div className="card-header">
          <SectionHeader title="Safety Status" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              label="Guardrail Pass"
              value={formatPct(safety?.guardrail_pass_rate)}
            />
            <StatCard
              label="Verify Failures"
              value={formatNum(safety?.verification_failures)}
            />
            <StatCard
              label="False Auto"
              value={formatNum(safety?.false_automation_count)}
            />
            <StatCard
              label="High Value Errors"
              value={formatNum(safety?.high_value_blocks)}
            />
          </div>
        </div>
      </div>

      {/* ─── System Components ────────────────────────────────────────────────── */}
      <div className="card mb-6">
        <div className="card-header">
          <SectionHeader title="System Components" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {SYSTEMS.map((s) => (
              <div
                key={s.name}
                className="flex items-center gap-2 p-2 rounded-lg border border-slate-100"
              >
                <span
                  className={`w-2.5 h-2.5 rounded-full ${
                    s.status ? "bg-emerald-500" : "bg-slate-300"
                  }`}
                />
                <span className="text-xs font-medium text-slate-600">
                  {s.name}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ─── Implemented Phases ───────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <SectionHeader
            title="Implemented Phases"
            subtitle={`${PHASES.length} phases`}
          />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {PHASES.map((p) => (
              <div
                key={p.num}
                className="flex items-center gap-3 p-2 rounded-lg hover:bg-slate-50"
              >
                <span className="text-emerald-500 text-sm">✅</span>
                <span className="text-xs font-semibold text-slate-500 w-8">
                  Phase {p.num}
                </span>
                <span className="text-sm text-slate-700">{p.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
