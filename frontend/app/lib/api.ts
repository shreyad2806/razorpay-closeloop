// ═══════════════════════════════════════════════════════════════════════════════
// Razorpay CloseLoop — Typed API Client
// Connects to the REAL backend. No invented endpoints.
// ═══════════════════════════════════════════════════════════════════════════════

import type {
  ApiResponse,
  HealthResponse,
  ExceptionListItem,
  ExceptionDetail,
  EvidenceResponse,
  SimilarCasesResponse,
  AnalysisResult,
  ExplanationResult,
  ResolveRequest,
  ResolveResponse,
  ApproveRequest,
  RejectRequest,
  EscalateRequest,
  ActionResponse,
  BatchItem,
  BatchSummary,
  FeedbackItem,
  FeedbackRequest,
  LearningMetrics,
  SystemMetrics,
  SafetyMetrics,
  ThroughputMetrics,
  ModelItem,
  ModelLineage,
} from "@/app/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─── Fetch wrapper with error handling ───────────────────────────────────────

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<{ data: T | null; error: string | null; ok: boolean }> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers || {}),
      },
      ...options,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => null);
      return {
        data: null,
        error: body?.error || `HTTP ${res.status}`,
        ok: false,
      };
    }

    const body = await res.json();
    return { data: body as T, error: null, ok: true };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : "Network error";
    return { data: null, error: msg, ok: false };
  }
}

async function apiPost<T>(
  path: string,
  body?: unknown
): Promise<{ data: T | null; error: string | null; ok: boolean }> {
  return apiFetch<T>(path, {
    method: "POST",
    body: body ? JSON.stringify(body) : undefined,
  });
}

// ─── Health ──────────────────────────────────────────────────────────────────

export async function getHealth() {
  return apiFetch<HealthResponse>("/health");
}

// ─── Exceptions ──────────────────────────────────────────────────────────────

export async function listExceptions(params?: {
  limit?: number;
  offset?: number;
  exception_type?: string;
  status?: string;
  risk_category?: string;
  batch_id?: string;
}) {
  const q = new URLSearchParams();
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  if (params?.exception_type) q.set("exception_type", params.exception_type);
  if (params?.status) q.set("status", params.status);
  if (params?.risk_category) q.set("risk_category", params.risk_category);
  if (params?.batch_id) q.set("batch_id", params.batch_id);
  const qs = q.toString();
  return apiFetch<ApiResponse<ExceptionListItem[]>>(
    `/exceptions${qs ? `?${qs}` : ""}`
  );
}

export async function getException(exceptionId: string) {
  return apiFetch<ApiResponse<ExceptionDetail>>(
    `/exceptions/${encodeURIComponent(exceptionId)}`
  );
}

export async function getEvidence(exceptionId: string) {
  return apiFetch<ApiResponse<EvidenceResponse>>(
    `/exceptions/${encodeURIComponent(exceptionId)}/evidence`
  );
}

export async function getSimilarCases(exceptionId: string, limit = 5) {
  return apiFetch<ApiResponse<SimilarCasesResponse>>(
    `/exceptions/${encodeURIComponent(exceptionId)}/similar?limit=${limit}`
  );
}

export async function analyzeException(exceptionId: string) {
  return apiPost<ApiResponse<AnalysisResult>>(
    `/exceptions/${encodeURIComponent(exceptionId)}/analyze`
  );
}

export async function explainException(exceptionId: string) {
  return apiFetch<ApiResponse<ExplanationResult>>(
    `/exceptions/${encodeURIComponent(exceptionId)}/explain`
  );
}

// ─── Resolution Actions ──────────────────────────────────────────────────────

export async function resolveException(
  exceptionId: string,
  req: ResolveRequest
) {
  return apiPost<ApiResponse<ResolveResponse>>(
    `/exceptions/${encodeURIComponent(exceptionId)}/resolve`,
    req
  );
}

export async function approveException(
  exceptionId: string,
  req: ApproveRequest
) {
  return apiPost<ApiResponse<ActionResponse>>(
    `/exceptions/${encodeURIComponent(exceptionId)}/approve`,
    req
  );
}

export async function rejectException(
  exceptionId: string,
  req: RejectRequest
) {
  return apiPost<ApiResponse<ActionResponse>>(
    `/exceptions/${encodeURIComponent(exceptionId)}/reject`,
    req
  );
}

export async function escalateException(
  exceptionId: string,
  req: EscalateRequest
) {
  return apiPost<ApiResponse<ActionResponse>>(
    `/exceptions/${encodeURIComponent(exceptionId)}/escalate`,
    req
  );
}

// ─── Batches ─────────────────────────────────────────────────────────────────

export async function listBatches() {
  return apiFetch<ApiResponse<BatchItem[]>>("/batches");
}

export async function getBatch(batchId: string) {
  return apiFetch<ApiResponse<BatchItem>>(
    `/batches/${encodeURIComponent(batchId)}`
  );
}

export async function createBatch(payload: {
  name: string;
  description?: string;
  num_merchants?: number;
  num_cases?: number;
}) {
  return apiPost<ApiResponse<BatchItem>>("/batches", payload);
}

export async function runBatch(batchId: string) {
  return apiPost<ApiResponse<BatchItem>>(
    `/batches/${encodeURIComponent(batchId)}/run`
  );
}

export async function getBatchSummary(batchId: string) {
  return apiFetch<ApiResponse<BatchSummary>>(
    `/batches/${encodeURIComponent(batchId)}/summary`
  );
}

// ─── Feedback / Learning ─────────────────────────────────────────────────────

export async function recordFeedback(req: FeedbackRequest) {
  return apiPost<ApiResponse<FeedbackItem>>("/feedback", req);
}

export async function getLearningMetrics() {
  return apiFetch<ApiResponse<LearningMetrics>>("/learning/metrics");
}

export async function getLearningDatasets() {
  return apiFetch<ApiResponse<{ total_examples: number }>>(
    "/learning/datasets"
  );
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

export async function getMetrics() {
  return apiFetch<ApiResponse<SystemMetrics>>("/metrics");
}

export async function getSafetyMetrics() {
  return apiFetch<ApiResponse<SafetyMetrics>>("/metrics/safety");
}

export async function getThroughputMetrics() {
  return apiFetch<ApiResponse<ThroughputMetrics>>("/metrics/throughput");
}

export async function getBatchMetrics(batchId: string) {
  return apiFetch<ApiResponse<Record<string, unknown>>>(
    `/metrics/batches/${encodeURIComponent(batchId)}`
  );
}

// ─── Models ──────────────────────────────────────────────────────────────────

export async function listModels() {
  return apiFetch<ApiResponse<ModelItem[]>>("/models");
}

export async function getModel(modelId: string) {
  return apiFetch<ApiResponse<ModelItem>>(
    `/models/${encodeURIComponent(modelId)}`
  );
}

export async function getModelLineage(modelId: string) {
  return apiFetch<ApiResponse<ModelLineage>>(
    `/models/${encodeURIComponent(modelId)}/lineage`
  );
}
