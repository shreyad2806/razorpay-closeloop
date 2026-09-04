"use client";

import { useEffect, useState } from "react";
import { listModels } from "@/app/lib/api";
import { Badge, LoadingState, EmptyState, CardSkeleton } from "@/components/ui";
import { formatPct } from "@/app/lib/utils";
import type { ModelItem } from "@/app/types";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      const res = await listModels();
      if (!mounted) return;
      if (res.ok && res.data?.data) {
        setModels(res.data.data as ModelItem[]);
        setError(null);
      } else if (!res.ok) {
        setError(res.error || "Failed to load models");
      } else {
        setModels([]);
      }
      setLoading(false);
    }
    load();
    return () => { mounted = false; };
  }, []);

  if (loading)
    return (
      <div>
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-slate-900">Models</h2>
          <p className="text-sm text-slate-400 mt-1">Loading…</p>
        </div>
        <div className="space-y-4">
          {Array.from({ length: 2 }).map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      </div>
    );

  if (error) {
    return (
      <div>
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-slate-900">Models</h2>
          <p className="text-sm text-slate-400 mt-1">
            MLflow model registry — Phase 10 model lineage and versioning
          </p>
        </div>
        <div className="card">
          <div className="card-body text-center py-12">
            <div className="text-red-500 text-4xl mb-3">⚠️</div>
            <h3 className="text-lg font-semibold text-slate-800 mb-1">Unable to load models</h3>
            <p className="text-sm text-slate-500">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-900">Models</h2>
        <p className="text-sm text-slate-400 mt-1">
          MLflow model registry — Phase 10 model lineage and versioning
        </p>
      </div>

      {models.length === 0 ? (
        <div className="card">
          <EmptyState icon="🤖" title="No models registered" />
        </div>
      ) : (
        <div className="space-y-4">
          {models.map((model) => (
            <div key={model.model_id} className="card">
              <div className="card-body">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-lg font-bold text-slate-800">
                        {model.model_name}
                      </h3>
                      <Badge text={model.status} variant="status" />
                    </div>
                    <div className="text-xs text-slate-400 mt-1 font-mono">
                      {model.model_id}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <MetricItem label="Version" value={model.model_version || "—"} />
                  <MetricItem
                    label="Precision"
                    value={model.precision != null ? formatPct(model.precision) : "—"}
                  />
                  <MetricItem
                    label="Recall"
                    value={model.recall != null ? formatPct(model.recall) : "—"}
                  />
                  <MetricItem
                    label="F1 Score"
                    value={model.f1 != null ? model.f1.toFixed(3) : "—"}
                  />
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricItem label="Dataset" value={model.dataset_version || "—"} />
                  <MetricItem label="Features" value={model.feature_version || "—"} />
                  <MetricItem
                    label="MLflow Run"
                    value={model.mlflow_run_id || "—"}
                    mono
                  />
                  <MetricItem
                    label="Created"
                    value={model.created_at || "—"}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MetricItem({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <div className="text-[10px] text-slate-400 uppercase tracking-wider mb-1">
        {label}
      </div>
      <div className={`text-sm font-medium text-slate-700 ${mono ? "font-mono text-xs" : ""}`}>
        {value}
      </div>
    </div>
  );
}
