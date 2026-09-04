"use client";

// ═══════════════════════════════════════════════════════════════════════════════
// EvidenceGraph — Interactive SVG evidence relationship visualization
// No external graph library. Pure SVG + CSS transforms for pan/zoom.
// ═══════════════════════════════════════════════════════════════════════════════

import { useRef, useState, useCallback, useMemo, useEffect } from "react";
import { formatPaise } from "@/app/lib/utils";
import type { EvidenceRecord } from "@/app/types";

// ─── Types ──────────────────────────────────────────────────────────────────

interface GraphNode {
  id: string;
  record: EvidenceRecord;
  x: number;
  y: number;
  tier: number;
}

interface GraphEdge {
  from: string;
  to: string;
  label: string;
}

interface ViewBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

// ─── Node type config ───────────────────────────────────────────────────────

const NODE_CONFIG: Record<
  string,
  { icon: string; color: string; bg: string; border: string; label: string; tier: number }
> = {
  PAYMENT: {
    icon: "💳",
    color: "#1e40af",
    bg: "#eff6ff",
    border: "#93c5fd",
    label: "Payment",
    tier: 0,
  },
  SETTLEMENT: {
    icon: "🏦",
    color: "#065f46",
    bg: "#ecfdf5",
    border: "#6ee7b7",
    label: "Settlement",
    tier: 1,
  },
  REFUND: {
    icon: "↩️",
    color: "#92400e",
    bg: "#fffbeb",
    border: "#fcd34d",
    label: "Refund",
    tier: 1,
  },
  FEE: {
    icon: "💰",
    color: "#6b21a8",
    bg: "#faf5ff",
    border: "#c4b5fd",
    label: "Fee",
    tier: 2,
  },
  TAX: {
    icon: "🏛️",
    color: "#9a3412",
    bg: "#fff7ed",
    border: "#fdba74",
    label: "Tax",
    tier: 2,
  },
  ADJUSTMENT: {
    icon: "⚖️",
    color: "#164e63",
    bg: "#ecfeff",
    border: "#67e8f9",
    label: "Adjustment",
    tier: 2,
  },
};

const NODE_W = 180;
const NODE_H = 72;
const TIER_GAP = 130;
const NODE_GAP_X = 40;
const SVG_W = 960;
const SVG_H = 520;

// ─── Graph layout ───────────────────────────────────────────────────────────

function layoutGraph(records: EvidenceRecord[]): {
  nodes: GraphNode[];
  edges: GraphEdge[];
} {
  if (records.length === 0) return { nodes: [], edges: [] };

  // Group by type
  const groups: Record<string, EvidenceRecord[]> = {};
  for (const r of records) {
    const t = r.record_type.toUpperCase();
    if (!groups[t]) groups[t] = [];
    groups[t].push(r);
  }

  // Determine tier widths
  const tiers: Record<number, string[]> = { 0: [], 1: [], 2: [] };
  for (const type of Object.keys(groups)) {
    const tier = NODE_CONFIG[type]?.tier ?? 2;
    if (!tiers[tier]) tiers[tier] = [];
    tiers[tier].push(type);
  }

  // Layout nodes per tier
  const nodes: GraphNode[] = [];
  const nodeMap: Record<string, GraphNode> = {};

  for (const [tierStr, types] of Object.entries(tiers)) {
    const tier = parseInt(tierStr);
    const tierY = 40 + tier * TIER_GAP;
    let totalW = 0;
    const typePositions: { type: string; w: number }[] = [];

    for (const type of types) {
      const count = groups[type].length;
      const w = count * NODE_W + (count - 1) * NODE_GAP_X;
      typePositions.push({ type, w });
      totalW += w;
    }
    totalW += (typePositions.length - 1) * (NODE_GAP_X * 1.5);

    let x = (SVG_W - totalW) / 2;

    for (const { type, w } of typePositions) {
      for (const rec of groups[type]) {
        const nodeW = NODE_W;
        const nx = x + nodeW / 2;
        const ny = tierY + NODE_H / 2;
        const node: GraphNode = {
          id: rec.record_id,
          record: rec,
          x: nx,
          y: ny,
          tier,
        };
        nodes.push(node);
        nodeMap[rec.record_id] = node;
      }
      x += w + NODE_GAP_X * 1.5;
    }
  }

  // Derive edges from record types
  const edges: GraphEdge[] = [];
  const paymentNodes = nodes.filter((n) => n.record.record_type === "PAYMENT");
  const settlementNodes = nodes.filter((n) => n.record.record_type === "SETTLEMENT");
  const refundNodes = nodes.filter((n) => n.record.record_type === "REFUND");
  const feeNodes = nodes.filter((n) => n.record.record_type === "FEE");
  const taxNodes = nodes.filter((n) => n.record.record_type === "TAX");
  const adjustmentNodes = nodes.filter((n) => n.record.record_type === "ADJUSTMENT");

  // Payment → Settlement
  for (const p of paymentNodes) {
    for (const s of settlementNodes) {
      edges.push({ from: p.id, to: s.id, label: "SETTLES" });
    }
  }
  // Payment → Refund
  for (const p of paymentNodes) {
    for (const r of refundNodes) {
      edges.push({ from: p.id, to: r.id, label: "REFUNDS" });
    }
  }
  // Payment → Fee
  for (const p of paymentNodes) {
    for (const f of feeNodes) {
      edges.push({ from: p.id, to: f.id, label: "CHARGES" });
    }
  }
  // Payment → Tax
  for (const p of paymentNodes) {
    for (const t of taxNodes) {
      edges.push({ from: p.id, to: t.id, label: "TAXES" });
    }
  }
  // Settlement → Adjustment
  for (const s of settlementNodes) {
    for (const a of adjustmentNodes) {
      edges.push({ from: s.id, to: a.id, label: "ADJUSTS" });
    }
  }
  // If no settlement, Payment → Adjustment directly
  if (settlementNodes.length === 0) {
    for (const p of paymentNodes) {
      for (const a of adjustmentNodes) {
        edges.push({ from: p.id, to: a.id, label: "ADJUSTS" });
      }
    }
  }

  return { nodes, edges };
}

// ─── SVG Edge ───────────────────────────────────────────────────────────────

function GraphEdgeView({
  from,
  to,
  label,
  highlighted,
  hoveredEdge,
  onHover,
}: {
  from: GraphNode;
  to: GraphNode;
  label: string;
  highlighted: boolean;
  hoveredEdge: string | null;
  onHover: (id: string | null) => void;
}) {
  const edgeId = `${from.id}->${to.id}`;
  const isHovered = hoveredEdge === edgeId;
  const opacity = highlighted ? 1 : 0.25;
  const strokeWidth = highlighted ? 2 : 1;

  // Calculate edge endpoints (bottom of source → top of target)
  const x1 = from.x;
  const y1 = from.y + NODE_H / 2;
  const x2 = to.x;
  const y2 = to.y - NODE_H / 2;

  // Control point for curve
  const midY = (y1 + y2) / 2;

  return (
    <g
      onMouseEnter={() => onHover(edgeId)}
      onMouseLeave={() => onHover(null)}
      style={{ cursor: "pointer" }}
    >
      <path
        d={`M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`}
        fill="none"
        stroke={highlighted ? "#6366f1" : "#cbd5e1"}
        strokeWidth={strokeWidth}
        opacity={opacity}
        strokeLinecap="round"
      />
      {/* Wider invisible path for hover target */}
      <path
        d={`M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`}
        fill="none"
        stroke="transparent"
        strokeWidth={16}
      />
      {isHovered && (
        <>
          <rect
            x={(x1 + x2) / 2 - 32}
            y={midY - 12}
            width={64}
            height={22}
            rx={6}
            fill="white"
            stroke="#e2e8f0"
            strokeWidth={1}
          />
          <text
            x={(x1 + x2) / 2}
            y={midY + 3}
            textAnchor="middle"
            fontSize={10}
            fontWeight={600}
            fill="#475569"
          >
            {label}
          </text>
        </>
      )}
    </g>
  );
}

// ─── SVG Node ───────────────────────────────────────────────────────────────

function GraphNodeView({
  node,
  selected,
  highlighted,
  hovered,
  onSelect,
  onHover,
}: {
  node: GraphNode;
  selected: boolean;
  highlighted: boolean;
  hovered: boolean;
  onSelect: () => void;
  onHover: (id: string | null) => void;
}) {
  const cfg = NODE_CONFIG[node.record.record_type.toUpperCase()] || NODE_CONFIG.PAYMENT;
  const x = node.x - NODE_W / 2;
  const y = node.y - NODE_H / 2;
  const dimmed = highlighted === false;

  return (
    <g
      onMouseEnter={() => onHover(node.id)}
      onMouseLeave={() => onHover(null)}
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
      style={{ cursor: "pointer" }}
    >
      {/* Shadow */}
      <rect
        x={x + 2}
        y={y + 2}
        width={NODE_W}
        height={NODE_H}
        rx={10}
        fill="rgba(0,0,0,0.06)"
      />
      {/* Background */}
      <rect
        x={x}
        y={y}
        width={NODE_W}
        height={NODE_H}
        rx={10}
        fill={selected ? "#eef2ff" : cfg.bg}
        stroke={selected ? "#6366f1" : hovered ? "#94a3b8" : cfg.border}
        strokeWidth={selected ? 2.5 : hovered ? 2 : 1.5}
        opacity={dimmed ? 0.35 : 1}
      />
      {/* Icon */}
      <text x={x + 14} y={y + 28} fontSize={20}>
        {cfg.icon}
      </text>
      {/* Type label */}
      <text
        x={x + 42}
        y={y + 22}
        fontSize={11}
        fontWeight={700}
        fill={cfg.color}
        opacity={dimmed ? 0.5 : 1}
      >
        {cfg.label}
      </text>
      {/* Record ID */}
      <text
        x={x + 42}
        y={y + 36}
        fontSize={10}
        fontFamily="ui-monospace, monospace"
        fill="#475569"
        opacity={dimmed ? 0.5 : 1}
      >
        {node.record.record_id}
      </text>
      {/* Amount */}
      <text
        x={x + 14}
        y={y + 58}
        fontSize={12}
        fontWeight={700}
        fontFamily="ui-monospace, monospace"
        fill="#1e293b"
        opacity={dimmed ? 0.5 : 1}
      >
        {formatPaise(node.record.amount_paise)}
      </text>
      {/* Status badge */}
      {node.record.status && (
        <text
          x={x + NODE_W - 14}
          y={y + 58}
          fontSize={9}
          textAnchor="end"
          fontWeight={600}
          fill={cfg.color}
          opacity={dimmed ? 0.5 : 0.7}
        >
          {node.record.status}
        </text>
      )}
      {/* Selected ring */}
      {selected && (
        <rect
          x={x - 3}
          y={y - 3}
          width={NODE_W + 6}
          height={NODE_H + 6}
          rx={12}
          fill="none"
          stroke="#6366f1"
          strokeWidth={2}
          strokeDasharray="6 3"
          opacity={0.6}
        />
      )}
    </g>
  );
}

// ─── Legend ──────────────────────────────────────────────────────────────────

function GraphLegend() {
  const types = ["PAYMENT", "SETTLEMENT", "REFUND", "FEE", "TAX", "ADJUSTMENT"];
  return (
    <div className="flex flex-wrap items-center gap-3 px-2 py-1.5">
      {types.map((t) => {
        const cfg = NODE_CONFIG[t];
        return (
          <div key={t} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-sm border"
              style={{ backgroundColor: cfg.bg, borderColor: cfg.border }}
            />
            <span className="text-[10px] text-slate-500 font-medium">
              {cfg.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Node Detail Panel ──────────────────────────────────────────────────────

function NodeDetailPanel({
  node,
  onClose,
}: {
  node: GraphNode;
  onClose: () => void;
}) {
  const cfg = NODE_CONFIG[node.record.record_type.toUpperCase()] || NODE_CONFIG.PAYMENT;
  return (
    <div
      className="absolute top-3 right-3 w-64 bg-white rounded-xl border border-slate-200 shadow-lg z-10 overflow-hidden"
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div
        className="px-4 py-3 flex items-center justify-between"
        style={{ backgroundColor: cfg.bg, borderBottom: `1px solid ${cfg.border}` }}
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">{cfg.icon}</span>
          <div>
            <div className="text-xs font-bold" style={{ color: cfg.color }}>
              {cfg.label}
            </div>
            <div className="text-[10px] font-mono text-slate-500">
              {node.record.record_id}
            </div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-600 text-sm font-bold w-6 h-6 flex items-center justify-center rounded hover:bg-white/50"
        >
          ✕
        </button>
      </div>
      {/* Details */}
      <div className="px-4 py-3 space-y-2.5">
        <DetailRow label="Amount" value={formatPaise(node.record.amount_paise)} bold />
        {node.record.status && (
          <DetailRow label="Status" value={node.record.status} />
        )}
        <DetailRow label="Type" value={cfg.label} />
        <DetailRow label="Record ID" value={node.record.record_id} mono />
      </div>
    </div>
  );
}

function DetailRow({
  label,
  value,
  bold,
  mono,
}: {
  label: string;
  value: string;
  bold?: boolean;
  mono?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[10px] text-slate-400 uppercase tracking-wider">
        {label}
      </span>
      <span
        className={`text-xs ${bold ? "font-bold text-slate-900" : "text-slate-700"} ${mono ? "font-mono" : ""}`}
      >
        {value}
      </span>
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────

export default function EvidenceGraph({
  evidence,
  loading,
  error,
}: {
  evidence: EvidenceRecord[] | null;
  loading?: boolean;
  error?: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [viewBox, setViewBox] = useState<ViewBox>({ x: 0, y: 0, w: SVG_W, h: SVG_H });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, vx: 0, vy: 0 });

  const { nodes, edges } = useMemo(
    () => layoutGraph(evidence || []),
    [evidence]
  );

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedId) || null,
    [nodes, selectedId]
  );

  // Connected node IDs for highlighting
  const connectedIds = useMemo(() => {
    if (!selectedId) return new Set<string>();
    const ids = new Set<string>([selectedId]);
    for (const e of edges) {
      if (e.from === selectedId) ids.add(e.to);
      if (e.to === selectedId) ids.add(e.from);
    }
    return ids;
  }, [selectedId, edges]);

  const hasHighlight = selectedId !== null;

  // ─── Zoom ───────────────────────────────────────────────────────────────

  const zoom = useCallback(
    (factor: number, cx?: number, cy?: number) => {
      setViewBox((vb) => {
        const centerX = cx ?? vb.x + vb.w / 2;
        const centerY = cy ?? vb.y + vb.h / 2;
        const newW = Math.max(200, Math.min(SVG_W * 2, vb.w * factor));
        const newH = Math.max(140, Math.min(SVG_H * 2, vb.h * factor));
        const ratio = newW / vb.w;
        return {
          x: centerX - (centerX - vb.x) * ratio,
          y: centerY - (centerY - vb.y) * ratio,
          w: newW,
          h: newH,
        };
      });
    },
    []
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const factor = e.deltaY > 0 ? 1.08 : 0.92;
      const cx = viewBox.x + ((e.clientX - rect.left) / rect.width) * viewBox.w;
      const cy = viewBox.y + ((e.clientY - rect.top) / rect.height) * viewBox.h;
      zoom(factor, cx, cy);
    },
    [viewBox, zoom]
  );

  // ─── Pan ────────────────────────────────────────────────────────────────

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      setIsPanning(true);
      panStart.current = {
        x: e.clientX,
        y: e.clientY,
        vx: viewBox.x,
        vy: viewBox.y,
      };
    },
    [viewBox]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isPanning) return;
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const dx = ((e.clientX - panStart.current.x) / rect.width) * viewBox.w;
      const dy = ((e.clientY - panStart.current.y) / rect.height) * viewBox.h;
      setViewBox((vb) => ({
        ...vb,
        x: panStart.current.vx - dx,
        y: panStart.current.vy - dy,
      }));
    },
    [isPanning, viewBox.w, viewBox.h]
  );

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  // ─── Reset view ─────────────────────────────────────────────────────────

  const resetView = useCallback(() => {
    setViewBox({ x: 0, y: 0, w: SVG_W, h: SVG_H });
    setSelectedId(null);
  }, []);

  // ─── Keyboard shortcuts ─────────────────────────────────────────────────

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelectedId(null);
      if (e.key === "+" || e.key === "=") zoom(0.9);
      if (e.key === "-") zoom(1.1);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [zoom]);

  // ─── Click background to deselect ──────────────────────────────────────

  const handleBgClick = useCallback(() => {
    setSelectedId(null);
  }, []);

  // ─── Empty / Loading / Error states ─────────────────────────────────────

  if (loading) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
            Evidence Graph
          </div>
          <div className="h-[320px] bg-slate-50 rounded-lg animate-pulse flex items-center justify-center">
            <span className="text-sm text-slate-400">Loading evidence graph…</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
            Evidence Graph
          </div>
          <div className="h-[320px] bg-red-50 rounded-lg flex flex-col items-center justify-center gap-2 border border-red-200">
            <span className="text-sm text-red-600 font-medium">
              Unable to load evidence graph
            </span>
            <span className="text-xs text-red-400">
              Check network connection and try again
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (!evidence || evidence.length === 0) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
            Evidence Graph
          </div>
          <div className="h-[320px] bg-slate-50 rounded-lg flex flex-col items-center justify-center gap-2">
            <span className="text-3xl opacity-30">📊</span>
            <span className="text-sm text-slate-400 font-medium">
              No evidence relationships available
            </span>
            <span className="text-xs text-slate-300">
              Evidence data will appear here when records are loaded
            </span>
          </div>
        </div>
      </div>
    );
  }

  // ─── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="card overflow-hidden">
      <div className="px-4 pt-4 pb-2 flex items-start justify-between">
        <div>
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">
            Evidence Graph
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">
            Trace the financial evidence behind this reconciliation decision
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => zoom(0.85)}
            className="w-7 h-7 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-600 text-sm font-bold flex items-center justify-center transition-colors"
            title="Zoom in"
          >
            +
          </button>
          <button
            onClick={() => zoom(1.18)}
            className="w-7 h-7 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-600 text-sm font-bold flex items-center justify-center transition-colors"
            title="Zoom out"
          >
            −
          </button>
          <button
            onClick={resetView}
            className="h-7 px-2 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-600 text-[10px] font-semibold flex items-center justify-center transition-colors"
            title="Reset view"
          >
            Reset
          </button>
        </div>
      </div>

      <GraphLegend />

      {/* Graph viewport */}
      <div
        ref={containerRef}
        className="relative overflow-hidden bg-[#f8fafc] border-t border-slate-100"
        style={{ height: 420, cursor: isPanning ? "grabbing" : "grab" }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Selected node detail panel */}
        {selectedNode && (
          <NodeDetailPanel
            node={selectedNode}
            onClose={() => setSelectedId(null)}
          />
        )}

        <svg
          width="100%"
          height="100%"
          viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
          preserveAspectRatio="xMidYMid meet"
          onClick={handleBgClick}
        >
          {/* Grid dots for spatial reference */}
          <defs>
            <pattern id="grid-dots" width="40" height="40" patternUnits="userSpaceOnUse">
              <circle cx="20" cy="20" r="0.8" fill="#e2e8f0" />
            </pattern>
          </defs>
          <rect
            x={viewBox.x - 200}
            y={viewBox.y - 200}
            width={viewBox.w + 400}
            height={viewBox.h + 400}
            fill="url(#grid-dots)"
          />

          {/* Edges */}
          {edges.map((e) => {
            const fromNode = nodes.find((n) => n.id === e.from);
            const toNode = nodes.find((n) => n.id === e.to);
            if (!fromNode || !toNode) return null;
            const isHighlighted = !hasHighlight || connectedIds.has(e.from);
            return (
              <GraphEdgeView
                key={`${e.from}-${e.to}`}
                from={fromNode}
                to={toNode}
                label={e.label}
                highlighted={isHighlighted}
                hoveredEdge={hoveredEdge}
                onHover={setHoveredEdge}
              />
            );
          })}

          {/* Nodes */}
          {nodes.map((n) => {
            const isSelected = n.id === selectedId;
            const isHighlighted = !hasHighlight || connectedIds.has(n.id);
            const isHovered = n.id === hoveredNodeId;
            return (
              <GraphNodeView
                key={n.id}
                node={n}
                selected={isSelected}
                highlighted={isHighlighted}
                hovered={isHovered}
                onSelect={() => setSelectedId(isSelected ? null : n.id)}
                onHover={setHoveredNodeId}
              />
            );
          })}
        </svg>

        {/* Coverage indicator */}
        <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-sm rounded-lg px-3 py-1.5 border border-slate-200">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span className="text-[10px] text-slate-500 font-medium">
              {nodes.length} nodes · {edges.length} relationships
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
