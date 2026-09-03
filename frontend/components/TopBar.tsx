"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/app/lib/api";

export function TopBar() {
  const [status, setStatus] = useState<"ok" | "error" | "checking">("checking");

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

  return (
    <div className="topbar">
      <div className="flex items-center gap-3">
        <span className="text-sm font-bold text-brand hidden sm:inline">
          Razorpay CloseLoop
        </span>
      </div>
      <div className="flex items-center gap-4 text-sm">
        <span className="flex items-center gap-1.5 text-slate-500">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              status === "ok"
                ? "bg-emerald-500"
                : status === "error"
                  ? "bg-red-500"
                  : "bg-amber-400 animate-pulse"
            }`}
          />
          Backend{" "}
          {status === "ok"
            ? "Connected"
            : status === "error"
              ? "Unavailable"
              : "Checking…"}
        </span>
        <span className="text-slate-300 hidden md:inline">|</span>
        <span className="text-slate-400 hidden md:inline">Production</span>
      </div>
    </div>
  );
}
