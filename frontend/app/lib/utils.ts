// ═══════════════════════════════════════════════════════════════════════════════
// Razorpay CloseLoop — Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

import { type ClassValue, clsx } from "clsx";

// Simple clsx implementation (no external dep needed)
export function cn(...inputs: (string | undefined | null | false)[]) {
  return inputs.filter(Boolean).join(" ");
}

// ─── Financial Formatting ────────────────────────────────────────────────────

/** Format paise (integer) to ₹ display */
export function formatPaise(paise: number | null | undefined): string {
  if (paise == null) return "—";
  const rupees = paise / 100;
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(rupees);
}

/** Format paise with sign */
export function formatPaiseSigned(paise: number | null | undefined): string {
  if (paise == null) return "—";
  if (paise === 0) return "₹0.00";
  const sign = paise > 0 ? "+" : "";
  return `${sign}${formatPaise(paise)}`;
}

/** Format percentage */
export function formatPct(value: number | null | undefined, decimals = 1): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Format number with locale */
export function formatNum(n: number | null | undefined): string {
  if (n == null) return "—";
  return new Intl.NumberFormat("en-IN").format(n);
}

/** Format date string */
export function fmtDate(s: string | null | undefined): string {
  if (!s) return "—";
  try {
    return new Date(s).toLocaleDateString("en-IN", {
      day: "numeric",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return s;
  }
}

// ─── Badge Helpers ───────────────────────────────────────────────────────────

export function riskBadge(risk: string): string {
  switch (risk?.toUpperCase()) {
    case "LOW":
      return "bg-emerald-100 text-emerald-800 border-emerald-200";
    case "MEDIUM":
      return "bg-amber-100 text-amber-800 border-amber-200";
    case "HIGH":
      return "bg-orange-100 text-orange-800 border-orange-200";
    case "CRITICAL":
      return "bg-red-100 text-red-800 border-red-200";
    default:
      return "bg-gray-100 text-gray-800 border-gray-200";
  }
}

export function statusBadge(status: string): string {
  switch (status?.toUpperCase()) {
    case "AUTO":
    case "RESOLVED":
    case "APPROVED":
      return "bg-emerald-100 text-emerald-800 border-emerald-200";
    case "PENDING":
    case "IN_PROGRESS":
      return "bg-amber-100 text-amber-800 border-amber-200";
    case "HUMAN_REVIEW":
      return "bg-blue-100 text-blue-800 border-blue-200";
    case "UNRESOLVED":
    case "REJECTED":
      return "bg-red-100 text-red-800 border-red-200";
    case "ESCALATED":
      return "bg-purple-100 text-purple-800 border-purple-200";
    default:
      return "bg-gray-100 text-gray-800 border-gray-200";
  }
}

export function guardrailBadge(decision: string): string {
  switch (decision?.toUpperCase()) {
    case "AUTO":
      return "bg-emerald-500 text-white";
    case "HUMAN_REVIEW":
      return "bg-blue-500 text-white";
    case "UNRESOLVED":
      return "bg-red-500 text-white";
    default:
      return "bg-gray-400 text-white";
  }
}

/** Format exception type for display */
export function formatExceptionType(t: string): string {
  return t
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Coverage badge color */
export function coverageBadge(c: string): string {
  switch (c) {
    case "FULLY_EXPLAINED":
      return "text-emerald-600";
    case "PARTIALLY_EXPLAINED":
      return "text-amber-600";
    case "CONFLICTING":
      return "text-red-600";
    default:
      return "text-gray-500";
  }
}
