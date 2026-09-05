"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Control Center", icon: "📊" },
  { href: "/exceptions", label: "Exceptions", icon: "⚠️" },
  { href: "/batches", label: "Batches", icon: "📦" },
  { href: "/learning", label: "Learning", icon: "🧠" },
  { href: "/models", label: "Models", icon: "🤖" },
  { href: "/system", label: "System", icon: "⚙️" },
];

export function Sidebar() {
  const pathname = usePathname();

  function isActive(href: string): boolean {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  }

  return (
    <aside className="sidebar" id="sidebar">
      <div className="sidebar-header">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-brand flex items-center justify-center text-white font-bold text-base shadow-sm">
            ₹
          </div>
          <div>
            <h1 className="text-base font-bold text-slate-900 tracking-tight leading-none">
              Razorpay <span className="text-brand">CloseLoop</span>
            </h1>
            <div className="subtitle text-[11px] text-slate-400 font-medium mt-1">
              Autonomous Financial Reconciliation
            </div>
          </div>
        </div>
      </div>
      <nav>
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={isActive(item.href) ? "active" : ""}
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </Link>
        ))}
      </nav>        <div className="sidebar-footer">
        <div>v1.0.0</div>
      </div>
    </aside>
  );
}
