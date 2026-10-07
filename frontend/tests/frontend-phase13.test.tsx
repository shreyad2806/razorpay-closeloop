/**
 * Phase 13 — executable frontend runtime tests.
 *
 * These render real React components against a stubbed `fetch` and assert on
 * DOM output. They are not source-inspection/documentation tests.
 */

import { Suspense } from "react";
import type { ReactNode } from "react";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
  AuthProvider,
  useAuth,
  hasPermission,
  PermissionGate,
  PERMISSION,
} from "@/app/lib/auth-context";
import { AuditTimeline } from "@/components/AuditTimeline";
import { ClosedLoopStages } from "@/components/ClosedLoopStages";
import {
  setAuthToken,
  clearAuthToken,
  getAuthToken,
} from "@/app/lib/api";
import type { CurrentPrincipal, ExceptionDetail } from "@/app/types";
import ExceptionDetailPage from "@/app/exceptions/[id]/page";
import {
  installFetchMock,
  envelope,
  VIEWER_PRINCIPAL,
  APPROVER_PRINCIPAL,
  ADMIN_PRINCIPAL,
  AUDIT_EVENTS,
  type FetchMock,
  type PrincipalResponse,
} from "./auth-helpers";

// ─── Next.js runtime stubs (jsdom has no app router) ─────────────────────────

jest.mock("next/navigation", () => ({
  useParams: () => ({ id: "exc-1" }),
  useSearchParams: () => new URLSearchParams(""),
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    back: jest.fn(),
    prefetch: jest.fn(),
  }),
}));

jest.mock("next/link", () => {
  const react = jest.requireActual("react") as typeof import("react");
  return {
    __esModule: true,
    default: ({
      href,
      children,
      ...rest
    }: {
      href?: unknown;
      children?: ReactNode;
      [key: string]: unknown;
    }) =>
      react.createElement(
        "a",
        { href: typeof href === "string" ? href : "#", ...rest },
        children
      ),
  };
});

// ─── Fixtures ────────────────────────────────────────────────────────────────

const EXCEPTION_DETAIL: ExceptionDetail = {
  exception_id: "exc-1",
  case_id: "case-1",
  merchant_id: "mer-1",
  payment_id: "pay-1",
  exception_type: "FEE_DIFFERENCE",
  expected_amount_paise: 100000,
  actual_amount_paise: 90000,
  difference_paise: 10000,
  risk_category: "HIGH",
  status: "HUMAN_REVIEW",
  resolution_type: "FEE_CORRECTION",
  adjustment_paise: 10000,
  workflow_id: "WF-1",
};

const NO_VIEW_AUDIT: PrincipalResponse = {
  subject: "restricted@example.com",
  actor_id: "p-restricted",
  roles: ["VIEWER"],
  permissions: ["view:exceptions"],
};

/** Probe component that exposes the full AuthProvider contract to the DOM. */
function SessionProbe() {
  const { isAuthenticated, principal, loading, error, roles, permissions, can, logout } =
    useAuth();

  if (loading) return <div data-testid="probe-loading">loading</div>;

  if (!isAuthenticated) {
    return (
      <div>
        <div data-testid="probe-unauthenticated">Not Authenticated</div>
        {error ? <div data-testid="probe-error">{error}</div> : null}
      </div>
    );
  }

  return (
    <div>
      <span data-testid="probe-subject">{principal?.subject}</span>
      <span data-testid="probe-actor">{principal?.actor_id}</span>
      <span data-testid="probe-roles">{roles.join(",")}</span>
      <span data-testid="probe-permissions">{permissions.join(",")}</span>
      <span data-testid="probe-can-approve">
        {String(can("approve:resolution"))}
      </span>
      <button onClick={logout}>Logout</button>
    </div>
  );
}

/** Renders the exception detail page inside AuthProvider with seeded data. */
async function renderExceptionPage(
  principal: PrincipalResponse | null
): Promise<FetchMock> {
  const mock = mockFetch ?? installFetchMock();
  mockFetch = mock;
  if (principal) {
    setAuthToken("test-token");
    mock.on("/auth/me", envelope(principal));
  } else {
    clearAuthToken();
  }
  // Order matters: the more specific routes must be registered first because
  // the helper matches on substring.
  mock.on(
    "/exceptions/exc-1/evidence",
    envelope({
      exception_id: "exc-1",
      evidence: [],
      total_amount_paise: 0,
      coverage: "FULLY_EXPLAINED",
      conflicts: [],
      missing_evidence: [],
      record_count: 0,
    })
  );
  mock.on(
    "/exceptions/exc-1/similar",
    envelope({ exception_id: "exc-1", similar_cases: [], count: 0 })
  );
  mock.on("/exceptions/exc-1", envelope(EXCEPTION_DETAIL));

  // The page suspends on `use(params)`; awaiting `act` flushes the resolved
  // params plus the seeded data fetches before assertions run.
  await act(async () => {
    render(
      <AuthProvider>
        <Suspense fallback={<div data-testid="suspense">loading…</div>}>
          <ExceptionDetailPage params={Promise.resolve({ id: "exc-1" })} />
        </Suspense>
      </AuthProvider>
    );
  });
  return mock;
}

let mockFetch: FetchMock | null = null;

function useFetchMock() {
  beforeEach(() => {
    mockFetch = installFetchMock();
  });
  afterEach(() => {
    mockFetch?.restore();
    mockFetch = null;
    clearAuthToken();
  });
}

// ═══════════════════════════════════════════════════════════════════════════
describe("Phase 13 — authentication", () => {
  useFetchMock();

  it("1. loads /auth/me and exposes principal, roles and permissions", async () => {
    setAuthToken("test-token");
    mockFetch!.on("/auth/me", envelope(VIEWER_PRINCIPAL as CurrentPrincipal));

    render(
      <AuthProvider>
        <SessionProbe />
      </AuthProvider>
    );

    await screen.findByTestId("probe-subject");
    expect(screen.getByTestId("probe-subject")).toHaveTextContent(
      "analyst@example.com"
    );
    expect(screen.getByTestId("probe-actor")).toHaveTextContent("p-analyst");
    expect(screen.getByTestId("probe-roles")).toHaveTextContent("ANALYST");
    expect(screen.getByTestId("probe-permissions")).toHaveTextContent(
      "view:audit"
    );
    expect(screen.getByTestId("probe-can-approve")).toHaveTextContent("false");

    // The request must carry the bearer token and nothing identity-bearing.
    expect(mockFetch!.callsFor("/auth/me")).toHaveLength(1);
    expect(mockFetch!.authHeaderFor("/auth/me")).toBe("Bearer test-token");
    // No token is ever rendered into the DOM.
    expect(screen.queryByText(/test-token/)).not.toBeInTheDocument();
  });

  it("2. reports unauthenticated when no token exists and does not call /auth/me", async () => {
    clearAuthToken();

    render(
      <AuthProvider>
        <SessionProbe />
      </AuthProvider>
    );

    await screen.findByTestId("probe-unauthenticated");
    expect(screen.getByTestId("probe-unauthenticated")).toHaveTextContent(
      "Not Authenticated"
    );
    expect(mockFetch!.callsFor("/auth/me")).toHaveLength(0);
  });

  it("3. handles an API failure by clearing the session and surfacing the error", async () => {
    setAuthToken("stale-token");
    mockFetch!.on(
      "/auth/me",
      { success: false, data: null, error: "authentication failed: expired" },
      401
    );

    render(
      <AuthProvider>
        <SessionProbe />
      </AuthProvider>
    );

    await screen.findByTestId("probe-unauthenticated");
    expect(screen.getByTestId("probe-error")).toHaveTextContent(
      "authentication failed"
    );
  });

  it("15. logout clears the session and the stored token", async () => {
    setAuthToken("test-token");
    mockFetch!.on("/auth/me", envelope(VIEWER_PRINCIPAL as CurrentPrincipal));

    render(
      <AuthProvider>
        <SessionProbe />
      </AuthProvider>
    );

    await screen.findByTestId("probe-subject");
    expect(getAuthToken()).toBe("test-token");

    await userEvent.click(screen.getByRole("button", { name: /logout/i }));

    expect(getAuthToken()).toBeNull();
    await screen.findByTestId("probe-unauthenticated");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
describe("Phase 13 — permission helper", () => {
  const viewer = VIEWER_PRINCIPAL as CurrentPrincipal;
  const approver = APPROVER_PRINCIPAL as CurrentPrincipal;

  it("4. returns true for a granted permission", () => {
    expect(hasPermission(viewer, "view:audit")).toBe(true);
    expect(hasPermission(approver, "approve:resolution")).toBe(true);
  });

  it("5. returns false for a missing permission", () => {
    expect(hasPermission(viewer, "approve:resolution")).toBe(false);
    expect(hasPermission(viewer, "nope:nothing")).toBe(false);
    expect(hasPermission(null, "view:audit")).toBe(false);
  });

  it("13. ADMIN does not automatically gain approve:resolution", () => {
    const admin = ADMIN_PRINCIPAL as CurrentPrincipal;
    expect(admin.roles).toContain("ADMIN");
    expect(hasPermission(admin, "approve:resolution")).toBe(false);
    expect(hasPermission(admin, "initiate:resolution")).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
describe("Phase 13 — permission-aware controls", () => {
  useFetchMock();

  it("6. hides the gated control when APPROVE_RESOLUTION is absent", async () => {
    setAuthToken("test-token");
    mockFetch!.on("/auth/me", envelope(VIEWER_PRINCIPAL as CurrentPrincipal));

    render(
      <AuthProvider>
        <PermissionGate
          permission={PERMISSION.APPROVE_RESOLUTION}
          fallback={<span data-testid="locked">permission required</span>}
        >
          <button>Approve Resolution</button>
        </PermissionGate>
      </AuthProvider>
    );

    await screen.findByTestId("locked");
    expect(
      screen.queryByRole("button", { name: /approve resolution/i })
    ).not.toBeInTheDocument();
  });

  it("7. shows the gated control when APPROVE_RESOLUTION is held", async () => {
    setAuthToken("test-token");
    mockFetch!.on("/auth/me", envelope(APPROVER_PRINCIPAL as CurrentPrincipal));

    render(
      <AuthProvider>
        <PermissionGate
          permission={PERMISSION.APPROVE_RESOLUTION}
          fallback={<span data-testid="locked">permission required</span>}
        >
          <button>Approve Resolution</button>
        </PermissionGate>
      </AuthProvider>
    );

    expect(
      await screen.findByRole("button", { name: /approve resolution/i })
    ).toBeInTheDocument();
    expect(screen.queryByTestId("locked")).not.toBeInTheDocument();
  });

  it("14. VIEW_AUDIT controls audit visibility", async () => {
    setAuthToken("test-token");
    mockFetch!.on("/auth/me", envelope(NO_VIEW_AUDIT as CurrentPrincipal));

    render(
      <AuthProvider>
        <PermissionGate
          permission={PERMISSION.VIEW_AUDIT}
          fallback={<span data-testid="audit-locked">no audit access</span>}
        >
          <AuditTimeline exceptionId="exc-1" />
        </PermissionGate>
      </AuthProvider>
    );

    await screen.findByTestId("audit-locked");
    expect(screen.queryByText(/audit trail/i)).not.toBeInTheDocument();
    expect(mockFetch!.callsFor("/audit")).toHaveLength(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
describe("Phase 13 — exception detail page RBAC wiring", () => {
  useFetchMock();

  it("disables Approve for a principal without approve:resolution", async () => {
    await renderExceptionPage(VIEWER_PRINCIPAL);

    const reviewTab = await screen.findByRole("button", { name: /^Review$/i });
    await userEvent.click(reviewTab);

    const approve = (await screen.findByTestId(
      "approve-button"
    )) as HTMLButtonElement;
    expect(approve).toBeDisabled();
    expect(screen.getByTestId("approve-locked")).toBeInTheDocument();

    // Reject requires reject:resolution — this principal lacks it too.
    const reject = screen.getByTestId("reject-button") as HTMLButtonElement;
    expect(reject).toBeDisabled();
  });

  it("enables Approve for an approver and keeps ADMIN locked", async () => {
    await renderExceptionPage(APPROVER_PRINCIPAL);
    await userEvent.click(await screen.findByRole("button", { name: /^Review$/i }));

    const approve = (await screen.findByTestId(
      "approve-button"
    )) as HTMLButtonElement;
    expect(approve).toBeEnabled();
    expect(screen.queryByTestId("approve-locked")).not.toBeInTheDocument();
  });

  it("keeps ADMIN locked on approval despite holding initiate:resolution", async () => {
    await renderExceptionPage(ADMIN_PRINCIPAL);
    await userEvent.click(await screen.findByRole("button", { name: /^Review$/i }));

    const approve = (await screen.findByTestId(
      "approve-button"
    )) as HTMLButtonElement;
    expect(approve).toBeDisabled();
    expect(screen.getByTestId("approve-locked")).toBeInTheDocument();
    // ADMIN may still initiate/escalate.
    const escalate = screen.getByTestId("escalate-button") as HTMLButtonElement;
    expect(escalate).toBeEnabled();
  });

  it("hides the Audit tab when view:audit is missing", async () => {
    await renderExceptionPage(NO_VIEW_AUDIT);

    await screen.findByRole("button", { name: /^Summary$/i });
    expect(
      screen.queryByRole("button", { name: /^Audit$/i })
    ).not.toBeInTheDocument();
    expect(mockFetch!.callsFor("/audit")).toHaveLength(0);
  });

  it("offers the Audit tab and fetches events when view:audit is held", async () => {
    // Must be registered before the base exception route: routes are matched
    // first-registered-wins on substring.
    mockFetch!.on("/exceptions/exc-1/audit", envelope(AUDIT_EVENTS));
    await renderExceptionPage(VIEWER_PRINCIPAL);

    await userEvent.click(await screen.findByRole("button", { name: /^Audit$/i }));

    expect(
      await screen.findByText("HUMAN_APPROVED")
    ).toBeInTheDocument();
    expect(screen.getByText(/ops_reviewer/)).toBeInTheDocument();
    expect(mockFetch!.callsFor("/audit").length).toBeGreaterThan(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
describe("Phase 13 — audit timeline", () => {
  useFetchMock();

  it("8. loads audit events rendered from the API response", async () => {
    setAuthToken("test-token");
    mockFetch!.on("/exceptions/exc-1/audit", envelope(AUDIT_EVENTS));

    render(
      <AuthProvider>
        <AuditTimeline exceptionId="exc-1" />
      </AuthProvider>
    );

    expect(await screen.findByText("RESOLUTION_PROPOSED")).toBeInTheDocument();
    expect(screen.getByText("HUMAN_APPROVED")).toBeInTheDocument();
    // Actor identity comes from the backend event, not from the frontend.
    expect(screen.getByText("ops_reviewer")).toBeInTheDocument();
    expect(screen.getByText("bot/guardrails")).toBeInTheDocument();
    // Decision and Outcome rows both carry the backend-supplied value.
    expect(screen.getAllByText("APPROVED").length).toBeGreaterThan(0);
    expect(screen.getByText(/HIGH/)).toBeInTheDocument();

    // Read-only: no mutation affordances are rendered.
    expect(
      screen.queryByRole("button", { name: /edit/i })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /delete/i })
    ).not.toBeInTheDocument();
  });

  it("9. renders the empty state when there is no audit history", async () => {
    setAuthToken("test-token");
    mockFetch!.on("/exceptions/exc-1/audit", envelope([]));

    render(
      <AuthProvider>
        <AuditTimeline exceptionId="exc-1" />
      </AuthProvider>
    );

    expect(await screen.findByText(/no audit events/i)).toBeInTheDocument();
  });

  it("10. renders the error state when the backend fails", async () => {
    setAuthToken("test-token");
    mockFetch!.on(
      "/exceptions/exc-1/audit",
      { success: false, data: null, error: "backend unavailable" },
      500
    );

    render(
      <AuthProvider>
        <AuditTimeline exceptionId="exc-1" />
      </AuthProvider>
    );

    expect(await screen.findByText(/backend unavailable/i)).toBeInTheDocument();
    expect(screen.queryByText(/no audit events/i)).not.toBeInTheDocument();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
describe("Phase 13 — execution vs verification", () => {
  it("11. displays execution and verification as separate stages", () => {
    render(<ClosedLoopStages status="VERIFYING" resolutionType="FEE_CORRECTION" />);

    const execution = screen.getByTestId("execution-stage");
    const verification = screen.getByTestId("verification-stage");

    expect(execution).toBeInTheDocument();
    expect(verification).toBeInTheDocument();
    expect(execution).not.toBe(verification);

    expect(screen.getByTestId("execution-state")).toHaveTextContent(
      "Action performed (backend reported)"
    );
    expect(screen.getByTestId("verification-state")).toHaveTextContent(
      "Verification pending"
    );
    // The two labels must never read the same thing.
    expect(screen.getByTestId("execution-state").textContent).not.toBe(
      screen.getByTestId("verification-state").textContent
    );
  });

  it("12. execution success never renders CLOSED", () => {
    const { unmount } = render(
      <ClosedLoopStages status="EXECUTING" resolutionType="FEE_CORRECTION" />
    );

    expect(screen.getByTestId("final-state")).toHaveTextContent("EXECUTING");
    expect(screen.getByTestId("not-closed-notice")).toBeInTheDocument();
    expect(screen.queryByText(/CLOSED/i)).not.toBeInTheDocument();
    unmount();

    render(<ClosedLoopStages status="APPROVED" />);
    expect(screen.getByTestId("final-state")).toHaveTextContent("APPROVED");
    expect(screen.queryByText(/CLOSED/i)).not.toBeInTheDocument();
  });

  it("renders CLOSED only when the backend already reports it", () => {
    render(<ClosedLoopStages status="CLOSED" />);

    expect(screen.getByTestId("final-state")).toHaveTextContent("CLOSED");
    expect(screen.getByTestId("verification-state")).toHaveTextContent(
      "Verified by backend"
    );
    expect(screen.queryByTestId("not-closed-notice")).not.toBeInTheDocument();
  });

  it("keeps execution SUCCESS + unverified state visibly unsettled", () => {
    render(<ClosedLoopStages status="RECONCILING" />);

    expect(screen.getByTestId("execution-state")).toHaveTextContent(
      "Action performed (backend reported)"
    );
    expect(screen.getByTestId("verification-state")).toHaveTextContent(
      "Verification pending"
    );
    expect(screen.getByTestId("not-closed-notice")).toBeInTheDocument();
    expect(screen.queryByText(/CLOSED/i)).not.toBeInTheDocument();
  });
});
