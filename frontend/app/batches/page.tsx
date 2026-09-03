"use client";

import { useEffect, useState, useCallback } from "react";
import {
  listBatches,
  createBatch,
  runBatch,
  getBatchSummary,
} from "@/app/lib/api";
import { Badge, LoadingState, EmptyState } from "@/components/ui";
import { formatPct, fmtDate } from "@/app/lib/utils";
import type { BatchItem, BatchSummary } from "@/app/types";

export default function BatchesPage() {
  const [batches, setBatches] = useState<BatchItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [running, setRunning] = useState<string | null>(null);
  const [summaries, setSummaries] = useState<Record<string, BatchSummary>>({});
  const [actionError, setActionError] = useState<string | null>(null);

  const loadBatches = useCallback(async () => {
    setLoading(true);
    setError(null);
    const { ok, data, error: apiError } = await listBatches();
    if (ok && data?.data) {
      const batchList = data.data as BatchItem[];
      setBatches(batchList);

      // Auto-load summaries for completed batches
      for (const batch of batchList) {
        if (batch.status === "COMPLETED" || batch.status === "PARTIAL") {
          const { ok: sOk, data: sData } = await getBatchSummary(batch.batch_id);
          if (sOk && sData?.data) {
            setSummaries((prev) => ({
              ...prev,
              [batch.batch_id]: sData.data as BatchSummary,
            }));
          }
        }
      }
    } else {
      setError(apiError || "Cannot connect to backend");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadBatches();
  }, [loadBatches]);

  async function handleCreate() {
    setCreating(true);
    setActionError(null);
    const { ok, error: apiError } = await createBatch({
      name: `Batch ${new Date().toISOString().slice(0, 10)}`,
      num_merchants: 5,
      num_cases: 20,
    });
    if (!ok) {
      setActionError(apiError || "Failed to create batch");
    }
    await loadBatches();
    setCreating(false);
  }

  async function handleRun(batchId: string) {
    setRunning(batchId);
    setActionError(null);
    const { ok, error: apiError } = await runBatch(batchId);
    if (!ok) {
      setActionError(apiError || "Failed to run batch");
    } else {
      // Load summary for the newly run batch
      const { ok: sOk, data: sData } = await getBatchSummary(batchId);
      if (sOk && sData?.data) {
        setSummaries((prev) => ({
          ...prev,
          [batchId]: sData.data as BatchSummary,
        }));
      }
    }
    await loadBatches();
    setRunning(null);
  }

  if (loading) return <LoadingState message="Loading batches…" />;

  if (error)
    return (
      <div>
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-slate-900">Batches</h2>
            <p className="text-sm text-slate-400 mt-1">
              Process and manage financial record batches
            </p>
          </div>
          <button className="btn btn-outline text-xs" onClick={loadBatches}>
            Retry
          </button>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-red-700 text-sm">
          <div className="font-semibold mb-1">Cannot load batches</div>
          <div>{error}</div>
        </div>
      </div>
    );

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Batches</h2>
          <p className="text-sm text-slate-400 mt-1">
            {batches.length === 0
              ? "No batches have been created yet"
              : `${batches.length} batch${batches.length !== 1 ? "es" : ""} total`}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            className="btn btn-outline text-xs"
            onClick={loadBatches}
            disabled={creating || running !== null}
          >
            Refresh
          </button>
          <button
            className="btn btn-primary"
            onClick={handleCreate}
            disabled={creating}
          >
            {creating ? "Creating…" : "+ Create Batch"}
          </button>
        </div>
      </div>

      {actionError && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-3 mb-4 text-sm text-red-700">
          {actionError}
        </div>
      )}

      {batches.length === 0 ? (
        <div className="card">
          <EmptyState
            icon="📦"
            title="No batches yet"
            description="Create your first batch to start processing financial records"
          />
        </div>
      ) : (
        <div className="space-y-4">
          {batches.map((batch) => {
            const summary = summaries[batch.batch_id];
            const isRunning = running === batch.batch_id;
            return (
              <div key={batch.batch_id} className="card">
                <div className="card-body">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-slate-800 font-mono">
                          {batch.batch_id}
                        </span>
                        <Badge text={batch.status} variant="status" />
                      </div>
                      <div className="text-xs text-slate-400 mt-1">
                        {batch.name || "Unnamed"} ·{" "}
                        {fmtDate(batch.created_at)}
                      </div>
                    </div>
                    {batch.status === "CREATED" && (
                      <button
                        className="btn btn-primary text-xs"
                        onClick={() => handleRun(batch.batch_id)}
                        disabled={isRunning}
                      >
                        {isRunning ? "Running…" : "Run"}
                      </button>
                    )}
                    {(batch.status === "COMPLETED" ||
                      batch.status === "PARTIAL") &&
                      !summary && (
                        <button
                          className="btn btn-outline text-xs"
                          onClick={async () => {
                            const { ok, data } = await getBatchSummary(
                              batch.batch_id
                            );
                            if (ok && data?.data) {
                              setSummaries((prev) => ({
                                ...prev,
                                [batch.batch_id]: data.data as BatchSummary,
                              }));
                            }
                          }}
                        >
                          Load Summary
                        </button>
                      )}
                  </div>

                  {summary && (
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-4 pt-4 border-t border-slate-100">
                      <SummaryItem
                        label="Exceptions"
                        value={String(summary.total_exceptions)}
                      />
                      <SummaryItem
                        label="Resolved"
                        value={String(summary.resolved)}
                        color="text-emerald-600"
                      />
                      <SummaryItem
                        label="Unresolved"
                        value={String(summary.unresolved)}
                        color="text-amber-600"
                      />
                      <SummaryItem
                        label="Escalated"
                        value={String(summary.escalated)}
                        color="text-purple-600"
                      />
                      <SummaryItem
                        label="Auto"
                        value={String(summary.auto_resolved)}
                        color="text-blue-600"
                      />
                    </div>
                  )}

                  {!summary && batch.status !== "CREATED" && (
                    <div className="text-xs text-slate-400 mt-3">
                      {batch.status === "RUNNING"
                        ? "Processing…"
                        : batch.status === "FAILED"
                          ? "Batch failed"
                          : ""}
                      {batch.exception_count
                        ? ` · ${batch.exception_count} exceptions`
                        : ""}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function SummaryItem({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div>
      <div className="text-[10px] text-slate-400 uppercase tracking-wider">
        {label}
      </div>
      <div className={`text-sm font-bold ${color || "text-slate-800"}`}>
        {value}
      </div>
    </div>
  );
}
