"use client";

import { useEffect, useState } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import {
  getLearningMetrics,
  getLearningDatasets,
  getSafetyMetrics,
  recordFeedback,
} from "@/app/lib/api";
import { StatCard, LoadingState, ErrorState, SectionHeader } from "@/components/ui";
import { formatPct, formatPaise, formatNum } from "@/app/lib/utils";
import type { LearningMetrics, SafetyMetrics } from "@/app/types";

const COLORS = ["#059669", "#dc2626", "#d97706", "#2563eb"];

export default function LearningPage() {
  const [metrics, setMetrics] = useState<LearningMetrics | null>(null);
  const [safety, setSafety] = useState<SafetyMetrics | null>(null);
  const [datasets, setDatasets] = useState<{ total_examples: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [feedbackResult, setFeedbackResult] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      const [mRes, sRes, dRes] = await Promise.all([
        getLearningMetrics(),
        getSafetyMetrics(),
        getLearningDatasets(),
      ]);
      if (!mounted) return;
      if (mRes.ok && mRes.data?.data) setMetrics(mRes.data.data as LearningMetrics);
      if (sRes.ok && sRes.data?.data) setSafety(sRes.data.data as SafetyMetrics);
      if (dRes.ok && dRes.data?.data) setDatasets(dRes.data.data as { total_examples: number });
      if (!mRes.ok) setError("Cannot load learning data");
      setLoading(false);
    }
    load();
    return () => { mounted = false; };
  }, []);

  async function handleSubmitFeedback(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    const { ok } = await recordFeedback({
      feedback_type: form.get("type") as "APPROVE" | "REJECT" | "CORRECT" | "ESCALATE",
      workflow_id: form.get("workflow_id") as string,
      reviewer: form.get("reviewer") as string,
    });
    setFeedbackResult(ok ? "Feedback recorded" : "Failed to record feedback");
    setTimeout(() => setFeedbackResult(null), 3000);
  }

  if (loading) return <LoadingState message="Loading learning metrics…" />;
  if (error) return <ErrorState title="Error" message={error} />;

  const a = metrics?.automation || {};
  const p = metrics?.precision || {};
  const h = metrics?.human_review || {};
  const r = metrics?.reward || {};
  const v = metrics?.verification || {};

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
          Phase 9 learning metrics and feedback analytics
        </p>
      </div>

      {/* ─── Metrics Grid ────────────────────────────────────────────────────── */}
      <div className="stat-grid mb-6">
        <StatCard
          label="Automation Rate"
          value={formatPct(a.automation_rate)}
          sub={`${formatNum(a.successful_auto)} / ${formatNum(a.total_exceptions)} resolved`}
        />
        <StatCard
          label="Precision"
          value={p.precision != null ? formatPct(p.precision) : "—"}
          sub="Correct auto-resolutions"
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
          sub={`${formatNum(r.total_rewards)} total`}
        />
        <StatCard
          label="Verification Pass"
          value={v.total_verified != null ? formatNum(v.total_verified) : "—"}
          sub={`${formatNum(v.total_executed)} executed`}
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
            <StatCard label="Pass Rate" value={formatPct(safety?.guardrail_pass_rate)} />
            <StatCard label="False Auto" value={formatNum(safety?.false_automation_count)} />
            <StatCard label="High Value" value={formatNum(safety?.high_value_blocks)} />
            <StatCard label="Conflict" value={formatNum(safety?.conflict_blocks)} />
            <StatCard label="Novelty" value={formatNum(safety?.novelty_blocks)} />
            <StatCard label="Verify Fail" value={formatNum(safety?.verification_failures)} />
          </div>
        </div>
      </div>

      {/* ─── Dataset ─────────────────────────────────────────────────────────── */}
      <div className="card mb-6">
        <div className="card-header">
          <SectionHeader title="Learning Dataset" />
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Total Examples" value={formatNum(datasets?.total_examples)} />
            <StatCard label="Safety Verdict" value={metrics?.safety?.verdict || "—"} />
            <StatCard label="Checks Passed" value={formatNum(metrics?.safety?.checks_passed)} />
            <StatCard label="Critical Failures" value={formatNum(metrics?.safety?.critical_failures?.length)} />
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
              {feedbackResult}
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
            <button type="submit" className="btn btn-primary">
              Submit Feedback
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
