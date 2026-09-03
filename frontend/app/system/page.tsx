"use client";

import { useEffect, useState } from "react";
import {
  getHealth,
  getMetrics,
  getSafetyMetrics,
  getThroughputMetrics,
  listModels,
  getLearningMetrics,
} from "@/app/lib/api";
import { StatCard, LoadingState, ErrorState, SectionHeader } from "@/components/ui";
import { formatPct, formatNum } from "@/app/lib/utils";
import type {
  HealthResponse,
  SystemMetrics,
  SafetyMetrics,
  ThroughputMetrics,
  ModelItem,
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

/** Component health status — derived from real backend API responses. */
type ComponentStatus = {
  name: string;
  label: "healthy" | "degraded" | "unavailable" | "unknown";
  source: string;
};

function deriveComponentStatuses(
  healthOk: boolean,
  modelsCount: number,
  hasMetrics: boolean,
  hasSafety: boolean,
  hasLearning: boolean,
  hasThroughput: boolean,
): ComponentStatus[] {
  return [
    {
      name: "Backend API",
      label: healthOk ? "healthy" : "unavailable",
      source: "GET /health",
    },
    {
      name: "Database",
      // If the API responds with data, the DB is reachable
      label: healthOk ? "healthy" : "unavailable",
      source: healthOk ? "inferred from API" : "no response",
    },
    {
      name: "ML Engine",
      // Models endpoint returning data means ML models are registered
      label: modelsCount > 0 ? "healthy" : healthOk ? "unknown" : "unavailable",
      source: modelsCount > 0 ? `GET /models (${modelsCount} models)` : "no model data",
    },
    {
      name: "Evidence Layer",
      // No dedicated health endpoint — only inferable from exceptions API working
      label: healthOk ? "unknown" : "unavailable",
      source: "no dedicated health endpoint",
    },
    {
      name: "LangGraph Agent",
      label: healthOk ? "unknown" : "unavailable",
      source: "no dedicated health endpoint",
    },
    {
      name: "Guardrails",
      // Safety metrics exist → guardrail system is available
      label: hasSafety ? "healthy" : healthOk ? "unknown" : "unavailable",
      source: hasSafety ? "GET /metrics/safety" : "no safety data",
    },
    {
      name: "Verification",
      // Metrics exist with verification data
      label: hasMetrics ? "healthy" : healthOk ? "unknown" : "unavailable",
      source: hasMetrics ? "GET /metrics" : "no metrics data",
    },
    {
      name: "Learning",
      label: hasLearning ? "healthy" : healthOk ? "unknown" : "unavailable",
      source: hasLearning ? "GET /learning/metrics" : "no learning data",
    },
    {
      name: "MCP",
      label: healthOk ? "unknown" : "unavailable",
      source: "no dedicated health endpoint",
    },
    {
      name: "LLM",
      label: healthOk ? "unknown" : "unavailable",
      source: "no dedicated health endpoint",
    },
  ];
}

export default function SystemPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [safety, setSafety] = useState<SafetyMetrics | null>(null);
  const [throughput, setThroughput] = useState<ThroughputMetrics | null>(null);
  const [modelsCount, setModelsCount] = useState(0);
  const [learningOk, setLearningOk] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      const [
        hRes, mRes, sRes, tRes, mlRes, lRes,
      ] = await Promise.all([
        getHealth(),
        getMetrics(),
        getSafetyMetrics(),
        getThroughputMetrics(),
        listModels(),
        getLearningMetrics(),
      ]);
      if (!mounted) return;
      if (hRes.ok && hRes.data) setHealth(hRes.data);
      if (mRes.ok && mRes.data?.data) setMetrics(mRes.data.data as SystemMetrics);
      if (sRes.ok && sRes.data?.data) setSafety(sRes.data.data as SafetyMetrics);
      if (tRes.ok && tRes.data?.data) setThroughput(tRes.data.data as ThroughputMetrics);
      if (mlRes.ok && mlRes.data?.data) setModelsCount((mlRes.data.data as ModelItem[]).length);
      if (lRes.ok) setLearningOk(true);
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
              value={
                (safety?.auto_decisions ?? 0) > 0
                  ? formatPct(safety?.guardrail_pass_rate)
                  : "—"
              }
              sub={(safety?.auto_decisions ?? 0) > 0 ? undefined : "No guardrail decisions yet"}
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
          <SectionHeader title="System Components" subtitle="Derived from real API responses" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {deriveComponentStatuses(
              health?.status === "ok",
              modelsCount,
              !!metrics,
              !!safety,
              learningOk,
              !!throughput,
            ).map((c) => (
              <div
                key={c.name}
                className="flex items-center gap-2 p-2 rounded-lg border border-slate-100"
                title={c.source}
              >
                <span
                  className={`w-2.5 h-2.5 rounded-full ${
                    c.label === "healthy"
                      ? "bg-emerald-500"
                      : c.label === "degraded"
                      ? "bg-amber-500"
                      : c.label === "unavailable"
                      ? "bg-red-500"
                      : "bg-slate-300"
                  }`}
                />
                <div className="flex flex-col">
                  <span className="text-xs font-medium text-slate-600">
                    {c.name}
                  </span>
                  <span className="text-[10px] text-slate-400">
                    {c.label}
                  </span>
                </div>
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
