"use client";

// ═══════════════════════════════════════════════════════════════════════════════
// Shared UI Components
// ═══════════════════════════════════════════════════════════════════════════════

import { riskBadge, statusBadge, guardrailBadge } from "@/app/lib/utils";

// ─── Stat Card ───────────────────────────────────────────────────────────────

export function StatCard({
  label,
  value,
  sub,
  className,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  className?: string;
}) {
  return (
    <div className={`stat-card ${className || ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

// ─── Badge ───────────────────────────────────────────────────────────────────

export function Badge({
  text,
  variant,
  className,
}: {
  text: string;
  variant?: "risk" | "status" | "guardrail" | "custom";
  className?: string;
}) {
  let colorClass = "";
  if (variant === "risk") colorClass = riskBadge(text);
  else if (variant === "status") colorClass = statusBadge(text);
  else if (variant === "guardrail") colorClass = guardrailBadge(text);
  else colorClass = className || "bg-slate-100 text-slate-700 border-slate-200";

  return <span className={`badge ${colorClass}`}>{text.replace(/_/g, " ")}</span>;
}

// ─── Loading State ───────────────────────────────────────────────────────────

export function LoadingState({ message }: { message?: string }) {
  return (
    <div className="loading-state">
      <div className="spinner" />
      <div className="text-sm">{message || "Loading…"}</div>
    </div>
  );
}

// ─── Empty State ─────────────────────────────────────────────────────────────

export function EmptyState({
  icon,
  title,
  description,
}: {
  icon: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">{icon}</div>
      <div className="text-sm font-medium text-slate-500">{title}</div>
      {description && (
        <div className="text-xs text-slate-400 mt-1">{description}</div>
      )}
    </div>
  );
}

// ─── Error State ─────────────────────────────────────────────────────────────

export function ErrorState({
  title,
  message,
}: {
  title: string;
  message?: string;
}) {
  return (
    <div className="error-state">
      <div className="font-semibold mb-1">{title}</div>
      {message && <div className="text-sm">{message}</div>}
    </div>
  );
}

// ─── Section Header ──────────────────────────────────────────────────────────

export function SectionHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between mb-4">
      <div>
        <h3 className="text-base font-bold text-slate-900">{title}</h3>
        {subtitle && (
          <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>
        )}
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}

// ─── Pipeline Progress ───────────────────────────────────────────────────────

const PIPELINE_STEPS = [
  "Records",
  "Reconcile",
  "Evidence",
  "Classify",
  "Similar",
  "Candidate",
  "Guardrails",
  "Decision",
  "Execute",
  "Verify",
];

export function PipelineProgress({ activeStep = 7 }: { activeStep?: number }) {
  return (
    <div className="pipeline">
      {PIPELINE_STEPS.map((step, i) => (
        <div key={step} className="flex items-center">
          <div
            className={`pipeline-step ${
              i < activeStep
                ? "done"
                : i === activeStep
                  ? "active"
                  : "pending"
            }`}
          >
            {i < activeStep && <span>✓</span>}
            <span>{step}</span>
          </div>
          {i < PIPELINE_STEPS.length - 1 && (
            <span className="pipeline-arrow">→</span>
          )}
        </div>
      ))}
    </div>
  );
}
