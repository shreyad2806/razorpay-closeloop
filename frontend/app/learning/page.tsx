"use client";

import { useEffect, useState } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import {
  getLearningMetrics,
  getLearningDatasets,
  getSafetyMetrics,
  recordFeedback,
} from "@/app/lib/api";
import { StatCard, StatCardSkeleton, CardSkeleton, LoadingState, ErrorState, SectionHeader, Toast } from "@/components/ui";
import { formatPct, formatPaise, formatNum } from "@/app/lib/utils";
import type { LearningMetrics, SafetyMetrics } from "@/app/types";

const COLORS = ["#059669", "#dc2626", "#d97706", "#2563eb"];

/** Show \"—\" when the backend has no learning data to compute a derived metric from. */
function rateOrDash(
  numerator: number | null | undefined,
  denominator: number | null | undefined
): string {
  if (!denominator || denominator === 0) return "—";
  return formatPct(numerator);
}

/** Whether the backend has executed at least some automation decisions. */
function hasAutomationData(m: LearningMetrics | null): boolean {
  return (m?.automation?.total_exceptions ?? 0) > 0;
}

/** Whether the backend has executed verification. */
function hasVerificationData(m: LearningMetrics | null): boolean {
  return (m?.verification?.total_executed ?? 0) > 0;
}

/** Whether the backend has run any safety checks. */
function hasSafetyCheckData(m: LearningMetrics | null): boolean {
  const checks = m?.safety?.checks ?? [];
  return checks.length > 0;
}

export default function LearningPage() {
  const [metrics, setMetrics] = useState<LearningMetrics | null>(null);
  const [safety, setSafety] = useState<SafetyMetrics | null>(null);
  const [datasets, setDatasets] = useState<{ total_examples: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [feedbackResult, setFeedbackResult] = useState<string | null>(null);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  async function loadAll() {
    const [mRes, sRes, dRes] = await Promise.all([
      getLearningMetrics(),
      getSafetyMetrics(),
      getLearningDatasets(),
    ]);
    if (mRes.ok && mRes.data?.data) setMetrics(mRes.data.data as LearningMetrics);
    if (sRes.ok && sRes.data?.data) setSafety(sRes.data.data as SafetyMetrics);
    if (dRes.ok && dRes.data?.data) setDatasets(dRes.data.data as { total_examples: number });
    return mRes.ok;
  }

  useEffect(() => {
    let mounted = true;
    async function init() {
      setLoading(true);
      const ok = await loadAll();
      if (!mounted) return;
      if (!ok) setError("Cannot load learning data");
      setLoading(false);
    }
    init();
    return () => { mounted = false; };
  }, []);

  async function handleSubmitFeedback(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setFeedbackLoading(true);
    setFeedbackError(null);
    setFeedbackResult(null);
    const form = new FormData(e.currentTarget);
    const { ok, error } = await recordFeedback({
      feedback_type: form.get("type") as "APPROVE" | "REJECT" | "CORRECT" | "ESCALATE",
      workflow_id: form.get("workflow_id") as string,
      reviewer: form.get("reviewer") as string,
    });
    if (ok) {
      setFeedbackResult("Feedback recorded");
      // Refresh learning metrics after feedback submission
      await loadAll();
    } else {
      setFeedbackError(error || "Failed to record feedback");
    }
    setFeedbackLoading(false);
    setTimeout(() => { setFeedbackResult(null); setFeedbackError(null); }, 5000);
  }

  if (loading)
    return (
      <div>
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-slate-900">Learning & Feedback</h2>
          <p className="text-sm text-slate-400 mt-1">Loading…</p>
        </div>
        <div className="stat-grid mb-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <StatCardSkeleton key={i} />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <CardSkeleton />
          <CardSkeleton />
        </div>
        <CardSkeleton />
      </div>
    );
  if (error) return <ErrorState title="Error" message={error} />;

  const a = metrics?.automation || {};
  const p = metrics?.precision || {};
  const h = metrics?.human_review || {};
  const r = metrics?.reward || {};
  const v = metrics?.verification || {};
  const s = metrics?.safety || {};
  const hasData = hasAutomationData(metrics);
  const hasVerif = hasVerificationData(metrics);
  const hasChecks = hasSafetyCheckData(metrics);

  const rewardData = [
    { name: "Positive", value: r.positive_rewards || 0 },
    { name: "Negative", value: r.negative_rewards || 0 },
  ].filter((d) => d.value > 0);

  const reviewData = [
    { name: "Approvals", value: h.human_approvals || 0 },
    { name: "Rejections", value: h.human_rejections || 0 },
    { name: "Corrections", value: h.human_corrections || 0 },
    { name: "Escalations", value: h.human_escalations || 0 },
  ].filter((d) => d.value > 0);

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-900">Learning & Feedback</h2>
        <p className="text-sm text-slate-400 mt-1">
          Feedback analytics and safety metrics
        </p>
      </div>

      {/* ─── Metrics Grid ────────────────────────────────────────────────────── */}
      <div className="stat-grid mb-6">
        <StatCard
          label="Automation Rate"
          value={hasData ? rateOrDash(a.automation_rate, a.total_exceptions) : "—"}
          sub={hasData ? `${formatNum(a.successful_auto)} / ${formatNum(a.total_exceptions)} resolved` : "No exceptions processed yet"}
        />
        <StatCard
          label="Precision"
          value={p.precision != null ? formatPct(p.precision) : "—"}
          sub={hasData ? "Correct auto-resolutions" : "No auto-resolutions to measure"}
        />
        <StatCard
          label="False Automation"
          value={formatNum(p.false_automation_count)}
          sub="Safety-critical metric"
        />
        <StatCard
          label="Human Review"
          value={formatNum(h.total_human_reviews)}
          sub="Total reviews"
        />
        <StatCard
          label="Average Reward"
          value={r.avg_reward != null ? r.avg_reward.toFixed(2) : "—"}
          sub={(r.total_rewards ?? 0) > 0 ? `${formatNum(r.total_rewards)} total` : "No reward data yet"}
        />
        <StatCard
          label="Verification Pass"
          value={hasVerif ? formatNum(v.total_verified) : "—"}
          sub={hasVerif ? `${formatNum(v.total_executed)} executed` : "No verifications run yet"}
        />
      </div>

      {/* ─── Charts ──────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Reward Distribution */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-sm font-bold text-slate-700">Reward Distribution</h3>
          </div>
          <div className="card-body">
            {rewardData.length > 0 ? (
              <div className="flex items-center gap-6">
                <ResponsiveContainer width="50%" height={160}>
                  <PieChart>
                    <Pie data={rewardData} dataKey="value" cx="50%" cy="50%" outerRadius={60}>
                      {rewardData.map((_, i) => (
                        <Cell key={i} fill={COLORS[i]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div className="space-y-2">
                  {rewardData.map((d, i) => (
                    <div key={d.name} className="flex items-center gap-2 text-xs">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i] }} />
                      <span className="text-slate-600">{d.name}</span>
                      <span className="font-semibold">{d.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-sm text-slate-400 text-center py-8">No reward data yet</div>
            )}
          </div>
        </div>

        {/* Human Review Breakdown */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-sm font-bold text-slate-700">Review Breakdown</h3>
          </div>
          <div className="card-body">
            {reviewData.length > 0 ? (
              <div className="flex items-center gap-6">
                <ResponsiveContainer width="50%" height={160}>
                  <PieChart>
                    <Pie data={reviewData} dataKey="value" cx="50%" cy="50%" outerRadius={60}>
                      {reviewData.map((_, i) => (
                        <Cell key={i} fill={COLORS[i]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div className="space-y-2">
                  {reviewData.map((d, i) => (
                    <div key={d.name} className="flex items-center gap-2 text-xs">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i] }} />
                      <span className="text-slate-600">{d.name}</span>
                      <span className="font-semibold">{d.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-sm text-slate-400 text-center py-8">No review data yet</div>
            )}
          </div>
        </div>
      </div>

      {/* ─── Safety Metrics ───────────────────────────────────────────────────── */}
      <div className="card mb-6">
        <div className="card-header">
          <SectionHeader title="Safety Metrics" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
            <StatCard label="Guardrail Pass Rate" value={hasData ? formatPct(safety?.guardrail_pass_rate) : "—"} sub={hasData ? undefined : "No guardrail decisions yet"} />
            <StatCard label="Auto Decisions" value={formatNum(safety?.auto_decisions)} />
            <StatCard label="High Value Blocks" value={formatNum(safety?.high_value_blocks)} />
            <StatCard label="Conflict Blocks" value={formatNum(safety?.conflict_blocks)} />
            <StatCard label="Novelty Blocks" value={formatNum(safety?.novelty_blocks)} />
            <StatCard label="Verify Failures" value={formatNum(safety?.verification_failures)} />
          </div>
        </div>
      </div>

      {/* ─── Dataset & Safety Verdict ─────────────────────────────────────────── */}
      <div className="card mb-6">
        <div className="card-header">
          <SectionHeader title="Learning Dataset & Safety Verdict" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              label="Total Examples"
              value={formatNum(datasets?.total_examples)}
              sub={datasets?.total_examples === 0 ? "No training data yet" : undefined}
            />
            <StatCard
              label="Safety Verdict"
              value={s?.verdict || "—"}
              sub={s?.verdict ? "From backend safety checks" : "No safety evaluation yet"}
            />
            <StatCard
              label="Checks Passed"
              value={hasChecks ? formatNum(s.checks_passed) : "—"}
              sub={hasChecks ? `${s.checks_passed} / ${(s.checks ?? []).length}` : "No checks executed yet"}
            />
            <StatCard
              label="Critical Failures"
              value={formatNum(s?.critical_failures?.length)}
            />
          </div>
        </div>
      </div>

      {/* ─── Feedback Form ───────────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <SectionHeader title="Record Feedback" />
        </div>
        <div className="card-body">
          {feedbackResult && (
            <div className="mb-3 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-lg px-3 py-2 text-sm">
              ✓ {feedbackResult}
            </div>
          )}
          {feedbackError && (
            <div className="mb-3 bg-red-50 text-red-700 border border-red-200 rounded-lg px-3 py-2 text-sm">
              ✗ {feedbackError}
            </div>
          )}
          <form onSubmit={handleSubmitFeedback} className="flex flex-wrap gap-3 items-end">
            <div>
              <label className="text-xs text-slate-400 uppercase tracking-wider block mb-1">Type</label>
              <select name="type" className="px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white">
                <option value="APPROVE">Approve</option>
                <option value="REJECT">Reject</option>
                <option value="CORRECT">Correct</option>
                <option value="ESCALATE">Escalate</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-400 uppercase tracking-wider block mb-1">Workflow ID</label>
              <input
                name="workflow_id"
                placeholder="workflow-id"
                required
                className="px-3 py-2 text-sm border border-slate-200 rounded-lg"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 uppercase tracking-wider block mb-1">Reviewer</label>
              <input
                name="reviewer"
                placeholder="your-id"
                className="px-3 py-2 text-sm border border-slate-200 rounded-lg"
              />
            </div>
            <button type="submit" className="btn btn-primary" disabled={feedbackLoading}>
              {feedbackLoading ? "Submitting…" : "Submit Feedback"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
