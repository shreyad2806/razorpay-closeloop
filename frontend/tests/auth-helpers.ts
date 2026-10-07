/**
 * Phase 13 test helpers — a real `globalThis.fetch` stub.
 *
 * The previous helper built `Response` objects and threw them away, so no
 * request was ever intercepted and component tests would have hit the network.
 * This one installs a route table, records every call (including the
 * Authorization header), and restores the original fetch afterwards.
 */

export type PrincipalResponse = {
  subject: string;
  actor_id: string;
  roles: string[];
  permissions: string[];
};

export type AuditResponse = {
  event_id: string;
  event_type: string;
  exception_id: string;
  workflow_id: string;
  actor: string;
  actor_type: string;
  timestamp: string | null;
  decision: string | null;
  confidence: number | null;
  risk: string | null;
  final_outcome: string | null;
  error: string | null;
  correction_of: string | null;
  correction_reason: string | null;
};

/** The `{ success, data }` envelope every CloseLoop endpoint returns. */
export interface Envelope<T> {
  success: boolean;
  data: T;
  error?: string | null;
  count?: number | null;
}

export function envelope<T>(data: T): Envelope<T> {
  return { success: true, data, error: null, count: null };
}

interface RecordedCall {
  url: string;
  method: string;
  headers: Record<string, string>;
}

interface Route {
  match: (url: string) => boolean;
  status: number;
  body: unknown;
}

export interface FetchMock {
  /** Register a route; first registered match wins. */
  on(path: string, body: unknown, status?: number): void;
  /** Every intercepted request, in order. */
  calls: RecordedCall[];
  /** Authorization header of the most recent call matching `path`. */
  authHeaderFor(path: string): string | undefined;
  callsFor(path: string): RecordedCall[];
  restore(): void;
}

/**
 * Replaces `globalThis.fetch` with a synchronous route table responder.
 * Returns plain `{ ok, status, json() }` objects so no `Response` global is
 * required inside jsdom.
 */
export function installFetchMock(): FetchMock {
  const original = globalThis.fetch;
  const routes: Route[] = [];
  const calls: RecordedCall[] = [];

  const impl = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : (input as Request).url;

    const rawHeaders = (init?.headers ?? {}) as Record<string, string>;
    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries(rawHeaders)) {
      headers[k.toLowerCase()] = String(v);
    }

    calls.push({ url, method: init?.method ?? "GET", headers });

    const route = routes.find((r) => r.match(url));
    if (!route) {
      return {
        ok: false,
        status: 404,
        json: async () => ({ success: false, error: `no mock for ${url}`, data: null }),
      };
    }
    return {
      ok: route.status >= 200 && route.status < 300,
      status: route.status,
      json: async () => route.body,
    };
  }) as unknown as typeof fetch;

  globalThis.fetch = impl;

  return {
    calls,
    on(path, body, status = 200) {
      routes.push({ match: (url) => url.includes(path), status, body });
    },
    authHeaderFor(path) {
      return calls.find((c) => c.url.includes(path))?.headers.authorization;
    },
    callsFor(path) {
      return calls.filter((c) => c.url.includes(path));
    },
    restore() {
      globalThis.fetch = original;
    },
  };
}

// ─── Fixtures ────────────────────────────────────────────────────────────────

/** ANALYST-equivalent: can read audit, cannot approve or reject. */
export const VIEWER_PRINCIPAL: PrincipalResponse = {
  subject: "analyst@example.com",
  actor_id: "p-analyst",
  roles: ["ANALYST"],
  permissions: ["view:exceptions", "view:audit", "investigate:exceptions"],
};

/** APPROVER-equivalent: holds the human approval boundary. */
export const APPROVER_PRINCIPAL: PrincipalResponse = {
  subject: "approver@example.com",
  actor_id: "p-approver",
  roles: ["APPROVER"],
  permissions: [
    "view:exceptions",
    "view:audit",
    "approve:resolution",
    "reject:resolution",
  ],
};

/**
 * ADMIN deliberately does NOT hold `approve:resolution` in the backend role
 * map, so this fixture must never unlock approval controls.
 */
export const ADMIN_PRINCIPAL: PrincipalResponse = {
  subject: "admin@example.com",
  actor_id: "p-admin",
  roles: ["ADMIN"],
  permissions: [
    "view:exceptions",
    "view:audit",
    "investigate:exceptions",
    "initiate:resolution",
    "request:execution",
    "admin:inspection",
    "admin:configuration",
  ],
};

export const AUDIT_EVENTS: AuditResponse[] = [
  {
    event_id: "evt-1",
    event_type: "RESOLUTION_PROPOSED",
    exception_id: "exc-1",
    workflow_id: "wf-1",
    actor: "bot/guardrails",
    actor_type: "AGENT",
    timestamp: "2026-09-18T10:00:00Z",
    decision: null,
    confidence: null,
    risk: null,
    final_outcome: null,
    error: null,
    correction_of: null,
    correction_reason: null,
  },
  {
    event_id: "evt-2",
    event_type: "HUMAN_APPROVED",
    exception_id: "exc-1",
    workflow_id: "wf-1",
    actor: "ops_reviewer",
    actor_type: "HUMAN",
    timestamp: "2026-09-18T10:30:00Z",
    decision: "APPROVED",
    confidence: null,
    risk: "HIGH",
    final_outcome: "APPROVED",
    error: null,
    correction_of: null,
    correction_reason: null,
  },
];
