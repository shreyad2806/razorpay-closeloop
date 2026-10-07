"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from "react";
import { getCurrentPrincipal, getAuthToken, clearAuthToken } from "./api";
import type { CurrentPrincipal } from "../types";

/**
 * Phase 13 — backend permission values.
 *
 * These are the exact ``Permission`` values emitted by
 * ``GET /auth/me`` (``app.auth.principal.Permission``). The frontend never
 * invents, derives or escalates permissions: it may only check membership of
 * the list the backend returned for the validated principal.
 *
 * Note: ADMIN deliberately does NOT hold APPROVE_RESOLUTION in the backend
 * role map, so the UI must never widen permissions based on a role name.
 */
export const PERMISSION = {
  VIEW_EXCEPTIONS: "view:exceptions",
  VIEW_EVIDENCE: "view:evidence",
  VIEW_RECONCILIATION: "view:reconciliation",
  VIEW_AUDIT: "view:audit",
  INVESTIGATE_EXCEPTIONS: "investigate:exceptions",
  GENERATE_PROPOSALS: "generate:proposals",
  INITIATE_RESOLUTION: "initiate:resolution",
  REQUEST_EXECUTION: "request:execution",
  APPROVE_RESOLUTION: "approve:resolution",
  REJECT_RESOLUTION: "reject:resolution",
} as const;

export type PermissionName =
  (typeof PERMISSION)[keyof typeof PERMISSION];

/**
 * Pure permission check over the backend-returned permission list.
 *
 * UX gating only — the backend RBAC layer remains authoritative.
 */
export function hasPermission(
  principal: CurrentPrincipal | null | undefined,
  permission: string
): boolean {
  if (!principal) return false;
  const perms = principal.permissions;
  if (!Array.isArray(perms)) return false;
  return perms.includes(permission);
}

interface AuthContextType {
  isAuthenticated: boolean;
  principal: CurrentPrincipal | null;
  loading: boolean;
  error: string | null;
  roles: string[];
  permissions: string[];
  can: (permission: string) => boolean;
  logout: () => void;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [principal, setPrincipal] = useState<CurrentPrincipal | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);

    // Session is driven purely by the token already held by the API client.
    // No identity, role or permission is ever supplied by the frontend.
    const token = getAuthToken();
    if (!token) {
      setIsAuthenticated(false);
      setPrincipal(null);
      setLoading(false);
      return;
    }

    const response = await getCurrentPrincipal();
    if (response.ok && response.data) {
      setIsAuthenticated(true);
      setPrincipal(response.data as CurrentPrincipal);
    } else {
      // 401 / 403 / network failure all collapse to "no usable session".
      setIsAuthenticated(false);
      setPrincipal(null);
      setError(response.error || "Failed to fetch principal");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const can = useCallback(
    (permission: string): boolean => hasPermission(principal, permission),
    [principal]
  );

  const logout = useCallback(() => {
    clearAuthToken();
    setIsAuthenticated(false);
    setPrincipal(null);
    setError(null);
    // Local session refresh only — this is not a Cognito/federated logout.
    window.location.reload();
  }, []);

  const roles = principal?.roles ?? [];
  const permissions = principal?.permissions ?? [];

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        principal,
        loading,
        error,
        roles,
        permissions,
        can,
        logout,
        refresh,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

/** Convenience wrapper around ``can`` — returns false when unauthenticated. */
export function usePermission(permission: string): boolean {
  const { can } = useAuth();
  return can(permission);
}

interface PermissionGateProps {
  /** Backend permission value (use the PERMISSION constants). */
  permission: string;
  children: ReactNode;
  /** Rendered instead of ``children`` when the permission is absent. */
  fallback?: ReactNode;
}

/**
 * Renders ``children`` only when the authenticated principal holds
 * ``permission``. Hiding controls is a convenience, never a security
 * boundary — the backend re-checks every operation.
 */
export function PermissionGate({
  permission,
  children,
  fallback = null,
}: PermissionGateProps) {
  const { can, loading } = useAuth();
  if (loading) return null;
  return <>{can(permission) ? children : fallback}</>;
}
