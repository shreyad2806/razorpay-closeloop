"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import { listExceptions } from "@/app/lib/api";
import { Badge, LoadingState, ErrorState, EmptyState } from "@/components/ui";
import { formatPaise, formatExceptionType, fmtDate } from "@/app/lib/utils";
import type { ExceptionListItem, ExceptionType, RiskCategory, ExceptionStatus } from "@/app/types";

const TYPES: (ExceptionType | "ALL")[] = [
  "ALL",
  "EXACT_MATCH",
  "FEE_DIFFERENCE",
  "REFUND_ADJUSTMENT",
  "TAX_ADJUSTMENT",
  "TIMING_DIFFERENCE",
  "PARTIAL_SETTLEMENT",
  "DUPLICATE",
  "MISSING_RECORD",
  "COMPLEX_MULTI_ADJUSTMENT",
  "UNKNOWN",
];

const STATUSES: (ExceptionStatus | "ALL")[] = [
  "ALL",
  "PENDING",
  "APPROVED",
  "REJECTED",
  "ESCALATED",
  "RESOLVED",
];

const RISKS: (RiskCategory | "ALL")[] = ["ALL", "LOW", "MEDIUM", "HIGH", "CRITICAL"];

export default function ExceptionsPage() {
  const [exceptions, setExceptions] = useState<ExceptionListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("ALL");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [riskFilter, setRiskFilter] = useState<string>("ALL");
  const [sortField, setSortField] = useState<"created_at" | "difference_paise" | "risk_category">("created_at");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      const { ok, data } = await listExceptions({ limit: 500 });
      if (!mounted) return;
      if (ok && data?.data) {
        setExceptions(data.data as ExceptionListItem[]);
      } else {
        setError("Cannot load exceptions");
      }
      setLoading(false);
    }
    load();
    return () => { mounted = false; };
  }, []);

  const filtered = useMemo(() => {
    let result = [...exceptions];
    if (typeFilter !== "ALL") result = result.filter((e) => e.exception_type === typeFilter);
    if (statusFilter !== "ALL") result = result.filter((e) => e.status === statusFilter);
    if (riskFilter !== "ALL") result = result.filter((e) => e.risk_category === riskFilter);
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(
        (e) =>
          e.exception_id.toLowerCase().includes(q) ||
          e.payment_id.toLowerCase().includes(q) ||
          e.merchant_id.toLowerCase().includes(q) ||
          e.exception_type.toLowerCase().includes(q) ||
          (e.batch_id || "").toLowerCase().includes(q)
      );
    }
    result.sort((a, b) => {
      let cmp = 0;
      if (sortField === "created_at") {
        cmp = (a.created_at || "").localeCompare(b.created_at || "");
      } else if (sortField === "difference_paise") {
        cmp = a.difference_paise - b.difference_paise;
      } else if (sortField === "risk_category") {
        const order = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
        cmp = (order[a.risk_category] ?? 4) - (order[b.risk_category] ?? 4);
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return result;
  }, [exceptions, typeFilter, statusFilter, riskFilter, search, sortField, sortDir]);

  if (loading) return <LoadingState message="Loading exceptions…" />;
  if (error) return <ErrorState title="Error" message={error} />;

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-900">Exceptions</h2>
        <p className="text-sm text-slate-400 mt-1">
          {exceptions.length} exceptions found across all batches
        </p>
      </div>

      {/* ─── Filters ────────────────────────────────────────────────────────── */}
      <div className="card mb-4">
        <div className="card-body">
          <div className="flex flex-wrap gap-3 items-center">
            <input
              type="text"
              placeholder="Search by ID, type, merchant…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="flex-1 min-w-[200px] px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand/20 focus:border-brand"
            />
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white"
            >
              {TYPES.map((t) => (
                <option key={t} value={t}>
                  {t === "ALL" ? "All Types" : formatExceptionType(t)}
                </option>
              ))}
            </select>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white"
            >
              {STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s === "ALL" ? "All Statuses" : s}
                </option>
              ))}
            </select>
            <select
              value={riskFilter}
              onChange={(e) => setRiskFilter(e.target.value)}
              className="px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white"
            >
              {RISKS.map((r) => (
                <option key={r} value={r}>
                  {r === "ALL" ? "All Risk Levels" : r}
                </option>
              ))}
            </select>
            {(typeFilter !== "ALL" || statusFilter !== "ALL" || riskFilter !== "ALL" || search) && (
              <button
                onClick={() => {
                  setSearch("");
                  setTypeFilter("ALL");
                  setStatusFilter("ALL");
                  setRiskFilter("ALL");
                }}
                className="btn btn-outline text-xs"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ─── Table ──────────────────────────────────────────────────────────── */}
      <div className="card">
        <div className="overflow-x-auto">
          {filtered.length === 0 ? (
            <EmptyState
              icon="🔍"
              title="No exceptions found"
              description="Try adjusting your filters"
            />
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Batch</th>
                  <th>Payment</th>
                  <th>Type</th>
                  <th
                    className="cursor-pointer hover:text-slate-700"
                    onClick={() => {
                      if (sortField === "difference_paise") setSortDir(sortDir === "asc" ? "desc" : "asc");
                      else { setSortField("difference_paise"); setSortDir("desc"); }
                    }}
                  >
                    Difference {sortField === "difference_paise" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th
                    className="cursor-pointer hover:text-slate-700"
                    onClick={() => {
                      if (sortField === "risk_category") setSortDir(sortDir === "asc" ? "desc" : "asc");
                      else { setSortField("risk_category"); setSortDir("asc"); }
                    }}
                  >
                    Risk {sortField === "risk_category" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th>Status</th>
                  <th
                    className="cursor-pointer hover:text-slate-700"
                    onClick={() => {
                      if (sortField === "created_at") setSortDir(sortDir === "asc" ? "desc" : "asc");
                      else { setSortField("created_at"); setSortDir("desc"); }
                    }}
                  >
                    Created {sortField === "created_at" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((exc) => (
                  <tr key={`${exc.exception_id}::${exc.batch_id || ''}`}>
                    <td>
                      <Link
                        href={`/exceptions/${exc.exception_id}?batch=${exc.batch_id || ''}`}
                        className="text-brand font-mono text-xs font-semibold hover:underline"
                      >
                        {exc.exception_id}
                      </Link>
                    </td>
                    <td className="text-xs text-slate-400 font-mono">
                      {exc.batch_id ? (
                        <span title={exc.batch_id}>
                          {exc.batch_id.length > 12 ? exc.batch_id.slice(0, 12) + "…" : exc.batch_id}
                        </span>
                      ) : (
                        <span className="text-slate-300">—</span>
                      )}
                    </td>
                    <td className="text-xs text-slate-500 font-mono">
                      {exc.payment_id}
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
        {filtered.length > 0 && (
          <div className="px-4 py-3 border-t border-slate-100 text-xs text-slate-400">
            Showing {filtered.length} of {exceptions.length} exceptions
          </div>
        )}
      </div>
    </div>
  );
}
