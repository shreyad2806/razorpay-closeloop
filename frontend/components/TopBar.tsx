"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/app/lib/api";
import { useAuth } from "@/app/lib/auth-context";

export function TopBar() {
  // Start with undefined so we can render nothing during SSR
  // and avoid hydration mismatch
  const [status, setStatus] = useState<"ok" | "error" | "checking" | null>(null);
  const { isAuthenticated, principal, loading, error, roles, permissions, logout } = useAuth();

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      const { ok } = await getHealth();
      if (mounted) setStatus(ok ? "ok" : "error");
    };
    check();
    const interval = setInterval(check, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // Render nothing during SSR to avoid hydration mismatch
  // The client will render the actual status after useEffect runs
  if (status === null) {
    return (
      <div className="topbar">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-slate-900 hidden sm:inline">
            Razorpay <span className="text-brand">CloseLoop</span>
          </span>
          <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 font-semibold hidden md:inline">
            Autonomous Financial Ops
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1.5 font-medium">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full bg-slate-300"
            />
            <span className="text-slate-500">Core Engine:</span>
            <span className="text-slate-400 font-semibold">—</span>
          </span>
          <span className="text-slate-200 hidden md:inline">|</span>
          <span className="text-slate-500 font-medium hidden md:inline">
            Financial Safety Guardrails Active
          </span>
          <span className="text-slate-200 hidden lg:inline">|</span>
          <span className="text-slate-400 hidden lg:inline">v1.0.0</span>
        </div>
      </div>
    );
  }

  return (
    <div className="topbar">
      <div className="flex items-center gap-3">
        <span className="text-sm font-bold text-slate-900 hidden sm:inline">
          Razorpay <span className="text-brand">CloseLoop</span>
        </span>
        <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 font-semibold hidden md:inline">
          Autonomous Financial Ops
        </span>
      </div>
      <div className="flex items-center gap-4 text-xs">
        <span className="flex items-center gap-1.5 font-medium">
          <span
            className={`inline-block w-2.5 h-2.5 rounded-full ${
              status === "ok"
                ? "bg-emerald-500 shadow-sm shadow-emerald-500/50"
                : status === "error"
                  ? "bg-rose-500"
                  : "bg-amber-400 animate-pulse"
            }`}
          />
          <span className="text-slate-500">Core Engine:</span>
          <span
            className={
              status === "ok"
                ? "text-emerald-700 font-semibold"
                : status === "error"
                  ? "text-rose-700 font-semibold"
                  : "text-amber-600 font-semibold"
            }
          >
            {status === "ok"
              ? "Live & Protected"
              : status === "error"
                ? "Offline"
                : "Checking…"}
          </span>
        </span>
        <span className="text-slate-200 hidden md:inline">|</span>
        <span className="text-slate-500 font-medium hidden md:inline">
          Financial Safety Guardrails Active
        </span>
        <span className="text-slate-200 hidden lg:inline">|</span>
        {/* Phase 13: authenticated identity, roles and permissions */}
        <span className="flex items-center gap-2 hidden md:inline">
          {loading ? (
            <span className="text-slate-400 font-medium">Checking session…</span>
          ) : !isAuthenticated ? (
            <span className="text-slate-500 font-medium" data-testid="auth-state">
              Not Authenticated
              {error ? <span className="text-rose-500 ml-1">({error})</span> : null}
            </span>
          ) : (
            <>
              <span
                className="text-slate-700 font-semibold"
                data-testid="auth-identity"
                title={principal?.subject}
              >
                {principal?.actor_id || principal?.subject || "user"}
              </span>
              {roles.length > 0 && (
                <span className="flex items-center gap-1" data-testid="auth-roles">
                  {roles.map((role) => (
                    <span
                      key={role}
                      className="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-600 font-semibold"
                    >
                      {role}
                    </span>
                  ))}
                </span>
              )}
              {permissions.length > 0 && (
                <span
                  className="flex flex-wrap items-center gap-1 max-w-[36%]"
                  data-testid="auth-permissions"
                  title={permissions.join(", ")}
                >
                  {permissions.map((perm) => (
                    <span
                      key={perm}
                      className="px-1.5 py-0.5 rounded bg-blue-50 border border-blue-200 text-blue-700 font-medium"
                    >
                      {perm}
                    </span>
                  ))}
                </span>
              )}
              <button
                onClick={logout}
                className="text-brand hover:underline font-medium"
                aria-label="logout"
              >
                Logout
              </button>
            </>
          )}
        </span>
        <span className="text-slate-200 hidden lg:inline">|</span>
        <span className="text-slate-400 hidden lg:inline">v1.0.0</span>
      </div>
    </div>
  );
}
