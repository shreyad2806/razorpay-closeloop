"use client";

import { useEffect, useState } from "react";
import { getExceptionAudit } from "@/app/lib/api";
import type { AuditEvent } from "@/app/types";

interface AuditTimelineProps {
  exceptionId: string;
}

export function AuditTimeline({ exceptionId }: AuditTimelineProps) {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      const response = await getExceptionAudit(exceptionId);
      if (!mounted) return;
      if (response.ok && response.data) {
        setEvents(response.data as AuditEvent[]);
      } else {
        setError(response.error || "Failed to load audit events");
      }
      setLoading(false);
    }
    load();
    return () => { mounted = false; };
  }, [exceptionId]);

  if (loading) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-sm font-bold text-slate-700">Audit Trail</h3>
        </div>
        <div className="card-body">
          <div className="text-sm text-slate-400 text-center py-8">Loading…</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-sm font-bold text-slate-700">Audit Trail</h3>
        </div>
        <div className="card-body">
          <div className="text-sm text-rose-600 text-center py-8">{error}</div>
        </div>
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-sm font-bold text-slate-700">Audit Trail</h3>
        </div>
        <div className="card-body">
          <div className="text-sm text-slate-400 text-center py-8">No audit events</div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-sm font-bold text-slate-700">Audit Trail</h3>
      </div>
      <div className="card-body">
        <div className="space-y-4">
          {events.map((event, idx) => (
            <div key={`${event.event_id}-${idx}`} className="flex gap-3 text-xs">
              <div className="flex flex-col items-center">
                <div className={`w-2 h-2 rounded-full ${
                  event.actor_type === "SYSTEM" ? "bg-slate-400" :
                  event.actor_type === "AGENT" ? "bg-blue-500" :
                  event.actor_type === "HUMAN" ? "bg-emerald-500" :
                  "bg-purple-500"
                }`} />
                {idx < events.length - 1 && <div className="w-0.5 h-full bg-slate-200" />}
              </div>
              <div className="flex-1 pb-4">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold text-slate-700">{event.event_type}</span>
                  <span className="text-slate-400">
                    {event.timestamp ? new Date(event.timestamp).toLocaleString() : "—"}
                  </span>
                </div>
                <div className="text-slate-600 mb-1">
                  <span className="font-medium">Actor:</span> {event.actor}
                  <span className="ml-2 px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                    {event.actor_type}
                  </span>
                </div>
                {event.decision && (
                  <div className="text-slate-600">
                    <span className="font-medium">Decision:</span> {event.decision}
                  </div>
                )}
                {event.final_outcome && (
                  <div className="text-slate-600">
                    <span className="font-medium">Outcome:</span> {event.final_outcome}
                  </div>
                )}
                {(event.risk || event.confidence != null) && (
                  <div className="text-slate-600">
                    {event.risk && (
                      <span className="mr-3">
                        <span className="font-medium">Risk:</span> {event.risk}
                      </span>
                    )}
                    {event.confidence != null && (
                      <span>
                        <span className="font-medium">Confidence:</span> {event.confidence}
                      </span>
                    )}
                  </div>
                )}
                {event.correction_of && (
                  <div className="text-slate-600">
                    <span className="font-medium">Corrects:</span> {event.correction_of}
                    {event.correction_reason ? ` — ${event.correction_reason}` : ""}
                  </div>
                )}
                {event.error && (
                  <div className="text-rose-600">
                    <span className="font-medium">Error:</span> {event.error}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
