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
        <h1>CloseLoop</h1>
        <div className="subtitle">Razorpay Financial Operations</div>
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
      </nav>
      <div className="sidebar-footer">
        <div>v1.0.0 · Phases 1–14</div>
      </div>
    </aside>
  );
}
