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

// ─── Skeleton Loaders ─────────────────────────────────────────────────────

function SkeletonLine({ className = "" }: { className?: string; style?: React.CSSProperties }) {
  return (
    <div
      className={`animate-pulse bg-slate-200 rounded ${className}`}
      aria-hidden="true"
    />
  );
}

export function StatCardSkeleton() {
  return (
    <div className="stat-card">
      <SkeletonLine className="h-3 w-20 mb-2" />
      <SkeletonLine className="h-7 w-16 mb-1" />
      <SkeletonLine className="h-3 w-24" />
    </div>
  );
}

export function TableSkeleton({
  rows = 5,
  columns = 6,
}: {
  rows?: number;
  columns?: number;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="data-table">
        <thead>
          <tr>
            {Array.from({ length: columns }).map((_, i) => (
              <th key={i}>
                <SkeletonLine className="h-3 w-16" />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, r) => (
            <tr key={r} className="!cursor-default">
              {Array.from({ length: columns }).map((_, c) => (
                <td key={c}>
                  <SkeletonLine className="h-4 w-3/4" />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function CardSkeleton() {
  return (
    <div className="card">
      <div className="card-body">
        <SkeletonLine className="h-5 w-40 mb-4" />
        <SkeletonLine className="h-4 w-full mb-2" />
        <SkeletonLine className="h-4 w-3/4 mb-2" />
        <SkeletonLine className="h-4 w-1/2" />
      </div>
    </div>
  );
}

export function DetailSkeleton() {
  return (
    <div className="space-y-6">
      <div className="card">
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i}>
                <SkeletonLine className="h-2 w-16 mb-2" />
                <SkeletonLine className="h-5 w-24" />
              </div>
            ))}
          </div>
        </div>
      </div>
      <CardSkeleton />
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div className="card">
      <div className="card-body">
        <SkeletonLine className="h-5 w-32 mb-4" />
        <div className="flex items-end gap-2 h-40">
          {[40, 65, 50, 80, 35, 70, 55].map((h, i) => (
            <SkeletonLine
              key={i}
              className="flex-1 rounded-t"
              style={{ height: `${h}%` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Toast ────────────────────────────────────────────────────────────────

export function Toast({
  message,
  type = "success",
  onDismiss,
}: {
  message: string;
  type?: "success" | "error" | "info";
  onDismiss?: () => void;
}) {
  const colors = {
    success: "bg-emerald-50 text-emerald-700 border-emerald-200",
    error: "bg-red-50 text-red-700 border-red-200",
    info: "bg-blue-50 text-blue-700 border-blue-200",
  };
  const icons = { success: "✓", error: "✕", info: "ℹ" };
  return (
    <div
      className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm font-medium ${colors[type]}`}
      role="alert"
    >
      <span className="text-base">{icons[type]}</span>
      <span className="flex-1">{message}</span>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-current opacity-50 hover:opacity-100 ml-2"
          aria-label="Dismiss"
        >
          ✕
        </button>
      )}
    </div>
  );
}
