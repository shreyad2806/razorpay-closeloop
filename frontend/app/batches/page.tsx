"use client";

import { useEffect, useState } from "react";
import { listBatches, createBatch, runBatch, getBatchSummary } from "@/app/lib/api";
import { Badge, LoadingState, EmptyState, SectionHeader } from "@/components/ui";
import { formatPct, fmtDate } from "@/app/lib/utils";
import type { BatchItem, BatchSummary } from "@/app/types";

export default function BatchesPage() {
  const [batches, setBatches] = useState<BatchItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [summaries, setSummaries] = useState<Record<string, BatchSummary>>({});

  useEffect(() => {
    loadBatches();
  }, []);

  async function loadBatches() {
    setLoading(true);
    const { ok, data } = await listBatches();
    if (ok && data?.data) {
      setBatches(data.data as BatchItem[]);
    } else {
      setError("Cannot load batches");
    }
    setLoading(false);
  }

  async function handleCreate() {
    setCreating(true);
    await createBatch({
      name: `Batch ${new Date().toISOString().slice(0, 10)}`,
      num_merchants: 5,
      num_cases: 20,
    });
    await loadBatches();
    setCreating(false);
  }

  async function handleRun(batchId: string) {
    await runBatch(batchId);
    const { ok, data } = await getBatchSummary(batchId);
    if (ok && data?.data) {
      setSummaries((prev) => ({ ...prev, [batchId]: data.data as BatchSummary }));
    }
    await loadBatches();
  }

  if (loading) return <LoadingState message="Loading batches…" />;
  if (error) return <EmptyState icon="📦" title="Cannot load batches" description={error} />;

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Batches</h2>
          <p className="text-sm text-slate-400 mt-1">
            Process and manage financial record batches
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={handleCreate}
          disabled={creating}
        >
          {creating ? "Creating…" : "+ Create Batch"}
        </button>
      </div>

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
                        {batch.name || "Unnamed"} · {fmtDate(batch.created_at)}
                      </div>
                    </div>
                    {batch.status === "CREATED" && (
                      <button
                        className="btn btn-primary text-xs"
                        onClick={() => handleRun(batch.batch_id)}
                      >
                        Run
                      </button>
                    )}
                  </div>

                  {summary && (
                    <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-100">
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase">Total</div>
                        <div className="text-sm font-bold">{summary.total_exceptions}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase">Auto</div>
                        <div className="text-sm font-bold text-emerald-600">{summary.auto_resolved}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase">Human</div>
                        <div className="text-sm font-bold text-blue-600">{summary.human_review}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase">Escalated</div>
                        <div className="text-sm font-bold text-purple-600">{summary.escalated}</div>
                      </div>
                    </div>
                  )}

                  {!summary && batch.status !== "CREATED" && (
                    <div className="text-xs text-slate-400 mt-3">
                      Processing… {batch.total_records ? `${batch.total_records} records` : ""}
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
