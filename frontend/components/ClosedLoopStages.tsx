"use client";

/**
 * Phase 13 — Execution vs Verification visibility.
 *
 * The whole point of this panel is that the two stages are NEVER collapsed:
 *
 *   Execution    = "the requested action was performed"
 *   Verification = "the financial state now matches the expected outcome"
 *
 * Every value rendered here comes from the backend exception record
 * (``GET /exceptions/{id}``). This component performs no arithmetic, no
 * reconciliation, no difference calculation, and no closure decision:
 * the final state badge is a verbatim echo of ``exception.status``.
 *
 * Backend contract (``app.domain.state_machines``):
 *   ... -> EXECUTING -> VERIFYING -> RECONCILING -> CLOSED
 *   CLOSED is reachable ONLY from RECONCILING. The frontend therefore never
 *   derives CLOSED — it can only display it when the backend already says so.
 */

type StageState = "pending" | "active" | "done" | "blocked";

export interface ClosedLoopStagesProps {
  /** Backend exception status — the single source of truth. */
  status: string;
  /** Backend-proposed resolution type, if one exists. */
  resolutionType?: string | null;
  /** Backend workflow id, if one exists. */
  workflowId?: string | null;
}

const EXECUTING_STATUSES = new Set(["EXECUTING"]);
const EXECUTED_STATUSES = new Set([
  "VERIFYING",
  "RECONCILING",
  "CLOSED",
  "RESOLVED",
]);
const EXECUTION_BLOCKED_STATUSES = new Set(["FAILED", "ROLLED_BACK"]);
const AWAITING_EXECUTION_STATUSES = new Set([
  "APPROVED",
  "AUTO_APPROVED",
  "HUMAN_REVIEW",
  "RESOLUTION_PROPOSED",
  "ANALYZED",
]);

const VERIFICATION_ACTIVE = new Set(["VERIFYING", "RECONCILING"]);
const VERIFICATION_DONE = new Set(["CLOSED"]);
const VERIFICATION_BLOCKED = new Set(["FAILED", "ROLLED_BACK"]);
const VERIFICATION_ESCALATED = new Set(["ESCALATED", "REJECTED"]);

function executionState(status: string): { state: StageState; label: string } {
  if (EXECUTING_STATUSES.has(status))
    return { state: "active", label: "Execution in progress" };
  if (EXECUTED_STATUSES.has(status))
    return { state: "done", label: "Action performed (backend reported)" };
  if (EXECUTION_BLOCKED_STATUSES.has(status))
    return { state: "blocked", label: "Execution failed / rolled back" };
  if (AWAITING_EXECUTION_STATUSES.has(status))
    return { state: "active", label: "Approved — awaiting execution" };
  if (status === "ESCALATED")
    return { state: "blocked", label: "Execution paused — escalated" };
  return { state: "pending", label: "No execution requested yet" };
}

function verificationState(status: string): {
  state: StageState;
  label: string;
} {
  if (VERIFICATION_ACTIVE.has(status))
    return { state: "active", label: "Verification pending" };
  if (VERIFICATION_DONE.has(status))
    return { state: "done", label: "Verified by backend" };
  if (VERIFICATION_BLOCKED.has(status))
    return { state: "blocked", label: "Verification failed" };
  if (VERIFICATION_ESCALATED.has(status))
    return { state: "blocked", label: "Not verified — routed for human review" };
  return { state: "pending", label: "Not verified" };
}

const STATE_STYLES: Record<StageState, string> = {
  pending: "border-slate-200 bg-slate-50 text-slate-500",
  active: "border-amber-300 bg-amber-50 text-amber-700",
  done: "border-emerald-300 bg-emerald-50 text-emerald-700",
  blocked: "border-rose-300 bg-rose-50 text-rose-700",
};

function StageChip({
  testId,
  state,
  label,
}: {
  testId: string;
  state: StageState;
  label: string;
}) {
  return (
    <span
      data-testid={testId}
      data-state={state}
      className={`inline-block px-2 py-1 rounded border text-[11px] font-semibold ${STATE_STYLES[state]}`}
    >
      {label}
    </span>
  );
}

/**
 * Renders the closed-loop pipeline as two visibly separate stages plus the
 * backend's final exception state.
 */
export function ClosedLoopStages({
  status,
  resolutionType,
  workflowId,
}: ClosedLoopStagesProps) {
  const exec = executionState(status);
  const ver = verificationState(status);

  return (
    <div className="card" data-testid="closed-loop-stages">
      <div className="card-header">
        <h3 className="text-sm font-bold text-slate-800">
          Execution &amp; Verification
        </h3>
        <p className="text-[11px] text-slate-400 mt-0.5">
          Two separate backend stages — executing an action does not verify it
        </p>
      </div>
      <div className="card-body space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* ── Stage 1: Execution ─────────────────────────────── */}
          <div
            className="rounded-lg border border-slate-200 p-3"
            data-testid="execution-stage"
          >
            <div className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">
              1 · Execution
            </div>
            <div className="text-xs font-semibold text-slate-700 mb-2">
              {resolutionType
                ? `Proposed: ${resolutionType.replace(/_/g, " ")}`
                : "No proposal submitted"}
            </div>
            <StageChip
              testId="execution-state"
              state={exec.state}
              label={exec.label}
            />
            <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
              Means the requested action was handed to the provider. It says
              nothing about the money.
            </p>
            {workflowId ? (
              <div className="text-[11px] text-slate-400 mt-1 font-mono">
                {workflowId}
              </div>
            ) : null}
          </div>

          {/* ── Stage 2: Verification ──────────────────────────── */}
          <div
            className="rounded-lg border border-slate-200 p-3"
            data-testid="verification-stage"
          >
            <div className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">
              2 · Verification
            </div>
            <div className="text-xs font-semibold text-slate-700 mb-2">
              Deterministic post-execution check
            </div>
            <StageChip
              testId="verification-state"
              state={ver.state}
              label={ver.label}
            />
            <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
              Only the backend decides whether expected and actual financial
              state match. The dashboard never computes this.
            </p>
          </div>

          {/* ── Stage 3: backend final state ───────────────────── */}
          <div
            className="rounded-lg border border-slate-200 p-3"
            data-testid="final-state-stage"
          >
            <div className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">
              3 · Final exception state
            </div>
            <div
              className="text-sm font-bold text-slate-900 break-words"
              data-testid="final-state"
            >
              {status}
            </div>
            <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
              Verbatim from <code className="font-mono">exception.status</code>.
              Closure is recorded by the backend only after deterministic
              verification — provider execution success alone never closes an
              exception.
            </p>
          </div>
        </div>

        {status !== "CLOSED" && (
          <div
            className="rounded-lg border border-amber-200 bg-amber-50/60 px-3 py-2 text-[11px] text-amber-800"
            data-testid="not-closed-notice"
          >
            This exception is still <strong>{status}</strong>. It is not
            financially settled until the backend reports the terminal state.
          </div>
        )}
      </div>
    </div>
  );
}

export default ClosedLoopStages;
