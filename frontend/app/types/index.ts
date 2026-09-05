// ═══════════════════════════════════════════════════════════════════════════════
// Razorpay CloseLoop — TypeScript Types
// Mirrors backend Pydantic schemas exactly. No invented types.
// ═══════════════════════════════════════════════════════════════════════════════

export type ExceptionType =
  | "EXACT_MATCH"
  | "FEE_DIFFERENCE"
  | "REFUND_ADJUSTMENT"
  | "TAX_ADJUSTMENT"
  | "TIMING_DIFFERENCE"
  | "PARTIAL_SETTLEMENT"
  | "DUPLICATE"
  | "MISSING_RECORD"
  | "COMPLEX_MULTI_ADJUSTMENT"
  | "UNKNOWN";

export type RiskCategory = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export type ResolutionType =
  | "REFUND_ADJUSTMENT"
  | "FEE_REVERSAL"
  | "SETTLEMENT_CORRECTION"
  | "NO_ACTION"
  | "UNKNOWN";

export type ExceptionStatus =
  | "PENDING"
  | "IN_PROGRESS"
  | "RESOLVED"
  | "ESCALATED"
  | "UNRESOLVED"
  | "APPROVED"
  | "REJECTED";

export type BatchStatus =
  | "CREATED"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "PARTIAL";

export type EvidenceCoverage =
  | "FULLY_EXPLAINED"
  | "PARTIALLY_EXPLAINED"
  | "UNEXPLAINED"
  | "CONFLICTING";

export type FeedbackType = "APPROVE" | "REJECT" | "CORRECT" | "ESCALATE";

export type ModelStatus =
  | "CANDIDATE"
  | "VALIDATION"
  | "PRODUCTION"
  | "ARCHIVED"
  | "REJECTED";

// ─── API Response Wrapper ────────────────────────────────────────────────────

export interface ApiResponse<T = unknown> {
  success: boolean;
  data: T;
  error?: string | null;
  count?: number | null;
}

export interface ErrorResponse {
  success: false;
  error: string;
  error_code: string;
  request_id?: string;
  details?: Record<string, unknown>;
}

// ─── Health ──────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  phases: string[];
}

// ─── Exception ───────────────────────────────────────────────────────────────

export interface ExceptionListItem {
  exception_id: string;
  case_id: string;
  merchant_id: string;
  payment_id: string;
  exception_type: ExceptionType;
  expected_amount_paise: number;
  actual_amount_paise: number;
  difference_paise: number;
  risk_category: RiskCategory;
  status: ExceptionStatus;
  classification_confidence?: number | null;
  resolvable?: boolean;
  batch_id?: string;
  created_at?: string | null;
}

export interface ExceptionDetail extends ExceptionListItem {
  detail?: boolean;
  guardrail_decision?: string | null;
  similar_case_count?: number;
  evidence_record_count?: number;
  updated_at?: string | null;
  resolution_type?: ResolutionType | string | null;
  adjustment_paise?: number | null;
  resolution_reason?: string | null;
  workflow_id?: string | null;
  candidate_id?: string | null;
  proposal_submitted_at?: string | null;
}

// ─── Evidence ────────────────────────────────────────────────────────────────

export interface EvidenceRecord {
  record_type: string;
  record_id: string;
  amount_paise: number;
  status?: string | null;
  metadata?: Record<string, unknown>;
}

export interface EvidenceResponse {
  exception_id: string;
  evidence: EvidenceRecord[];
  total_amount_paise: number;
  coverage: EvidenceCoverage;
  conflicts: string[];
  missing_evidence: string[];
  payment_id?: string;
  record_count?: number;
}

// ─── Similar Cases ───────────────────────────────────────────────────────────

export interface SimilarCaseItem {
  case_id: string;
  exception_type: ExceptionType;
  similarity_score: number;
  resolution_type?: ResolutionType | null;
  adjustment_paise?: number | null;
  risk_category?: RiskCategory | null;
}

export interface SimilarCasesResponse {
  exception_id: string;
  similar_cases: SimilarCaseItem[];
  count: number;
  confidence?: string | null;
  total_candidates?: number;
}

// ─── Resolution Candidate ────────────────────────────────────────────────────

export interface ResolutionCandidate {
  resolution_type: string;
  adjustment_paise: number;
  confidence: number;
  source: string;
  description?: string;
  evidence_compatible?: boolean;
}

// ─── Guardrail ───────────────────────────────────────────────────────────────

export interface GuardrailSummary {
  decision: string;
  confidence: number;
  risk_category: RiskCategory;
  reasons: string[];
  exposure_paise: number;
}

// ─── Analysis ────────────────────────────────────────────────────────────────

export interface FinancialDiscrepancy {
  expected_amount_paise: number;
  actual_amount_paise: number;
  difference_paise: number;
  exception_type: ExceptionType;
}

export interface EvidenceSummary {
  record_count: number;
  coverage: EvidenceCoverage;
  explained_amount_paise?: number;
  remaining_difference_paise?: number;
  conflicts: string[];
  missing_evidence: string[];
}

export interface AnalysisResult {
  exception_id: string;
  case_id?: string;
  financial_discrepancy: FinancialDiscrepancy;
  evidence: EvidenceSummary;
  classification_type?: ExceptionType;
  classification_confidence?: number | null;
  similar_case_count?: number;
  similar_case_summary?: string;
  candidates: ResolutionCandidate[];
  selected_candidate?: string;
  risk?: string;
  ml_confidence?: number | null;
  guardrail: GuardrailSummary;
  ai_explanation?: string;
  ai_uncertainty?: string;
  llm_provider?: string;
  llm_model?: string;
  fallback_used?: boolean;
}

// ─── Explanation ─────────────────────────────────────────────────────────────

export interface ExplanationResult {
  exception_id: string;
  case_id?: string;
  summary: string;
  reason: string;
  evidence_summary: string;
  uncertainty: string;
  limitations?: string;
  expected_amount_paise?: number;
  actual_amount_paise?: number;
  difference_paise?: number;
  exception_type?: ExceptionType;
  evidence_record_count?: number;
  evidence_coverage?: EvidenceCoverage;
  conflicts?: string[];
  missing_evidence?: string[];
  llm_provider?: string;
  llm_model?: string;
  fallback_used?: boolean;
}

// ─── Resolution Actions ──────────────────────────────────────────────────────

export interface ResolveRequest {
  resolution_type: ResolutionType;
  adjustment_paise: number;
  reason?: string;
  candidate_id?: string;
}

export interface ResolveResponse {
  exception_id: string;
  resolution_type: ResolutionType;
  status: string;
  adjustment_paise: number;
  guardrail_decision: string | null;
  execution_result: string | null;
  verification_result: string | null;
  workflow_id?: string;
  message?: string;
}

export interface ApproveRequest {
  approved_by: string;
  comments?: string;
}

export interface RejectRequest {
  rejected_by: string;
  reason: string;
}

export interface EscalateRequest {
  reason: string;
  escalated_by?: string;
  priority?: string;
}

export interface ActionResponse {
  exception_id: string;
  status: string;
  approved_by?: string;
  rejected_by?: string;
  escalated_by?: string;
  reason?: string;
  feedback_id?: string;
  message?: string;
}

// ─── Batch ───────────────────────────────────────────────────────────────────

export interface BatchItem {
  batch_id: string;
  name?: string;
  status: BatchStatus;
  created_at?: string;
  total_records?: number;
  matched_records?: number;
  exception_count?: number;
  processing_time_ms?: number;
  success_count?: number;
  failure_count?: number;
}

export interface BatchSummary {
  batch_id: string;
  status: BatchStatus;
  total_exceptions: number;
  resolved: number;
  unresolved: number;
  escalated: number;
  auto_resolved: number;
  human_review: number;
  verification_passed: number;
  verification_failed: number;
  financial_impact_paise: number;
}

// ─── Feedback ────────────────────────────────────────────────────────────────

export interface FeedbackItem {
  feedback_id: string;
  workflow_id: string;
  exception_id?: string;
  feedback_type: FeedbackType;
  reviewer: string;
  system_prediction?: string;
  created_at?: string;
}

export interface FeedbackRequest {
  feedback_type: FeedbackType;
  workflow_id: string;
  exception_id?: string;
  candidate_id?: string;
  reviewer?: string;
}

// ─── Learning Metrics ────────────────────────────────────────────────────────

export interface LearningMetrics {
  metrics_id?: string;
  automation: {
    total_exceptions?: number;
    automation_rate?: number;
    human_review_rate?: number;
    unresolved_rate?: number;
    successful_auto?: number;
    failed_auto?: number;
  };
  precision: {
    correct_auto?: number;
    incorrect_auto?: number;
    precision?: number | null;
    false_automation_count?: number;
    false_automation_rate?: number | null;
  };
  human_review: {
    total_human_reviews?: number;
    human_corrections?: number;
    human_rejections?: number;
    human_approvals?: number;
    human_escalations?: number;
  };
  reward: {
    total_rewards?: number;
    avg_reward?: number | null;
    positive_rewards?: number;
    negative_rewards?: number;
  };
  financial: {
    total_adjustment_paise?: number;
    total_error_impact_paise?: number;
    high_value_error_count?: number;
    discrepancy_eliminated_count?: number;
  };
  verification: {
    total_executed?: number;
    total_verified?: number;
    total_rolled_back?: number;
    total_verification_failed?: number;
  };
  safety: {
    verdict?: string;
    checks?: unknown[];
    checks_passed?: number;
    checks_failed?: number;
    critical_failures?: string[];
  };
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

export interface SystemMetrics {
  total_records: number;
  matched_records: number;
  exceptions: number;
  match_rate: number;
  exception_rate: number;
  automation_rate: number;
  human_review: number;
  human_review_rate: number;
  unresolved: number;
  unresolved_rate: number;
  auto_resolved: number;
  verification_passed: number;
  verification_failed: number;
  financial_impact_paise: number;
}

export interface SafetyMetrics {
  auto_decisions: number;
  human_review_decisions: number;
  unresolved_decisions: number;
  guardrail_blocks: number;
  high_value_blocks: number;
  conflict_blocks: number;
  novelty_blocks: number;
  verification_failures: number;
  guardrail_pass_rate: number;
  false_automation_count?: number;
}

export interface ThroughputMetrics {
  total_records_processed: number;
  total_processing_time_ms: number;
  avg_processing_time_ms: number;
  records_per_second: number;
  batches_processed: number;
}

// ─── Model ───────────────────────────────────────────────────────────────────

export interface ModelItem {
  model_id: string;
  model_name: string;
  model_version: string;
  status: ModelStatus;
  mlflow_run_id?: string | null;
  dataset_version?: string | null;
  feature_version?: string | null;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  created_at?: string | null;
  promoted_at?: string | null;
}

export interface ModelLineage {
  model_id: string;
  model_version: string;
  mlflow_run_id?: string | null;
  dataset_version?: string | null;
  feature_version?: string | null;
  training_config?: Record<string, unknown>;
  metrics?: Record<string, number>;
  artifacts?: string[];
}
