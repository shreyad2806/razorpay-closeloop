"""
Tests for Phase 13.2 — API Schema Validation.

Tests all Pydantic request/response schemas for:
- Valid input
- Missing required fields
- Invalid fields
- Invalid enums
- Invalid amounts
- Invalid confidence values
- Empty values
- Oversized values
- Malformed IDs
"""

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AnalysisDepth,
    AnalysisResult,
    ApproveRequest,
    ApiResponse,
    BatchCreateRequest,
    BatchMetrics,
    BatchResponse,
    BatchStatus,
    BatchSummaryResponse,
    DatasetInfo,
    EscalateRequest,
    EvidenceCoverage,
    EvidenceRecord,
    EvidenceResponse,
    EvidenceSummary,
    ExceptionDetail,
    ExceptionListItem,
    ExceptionStatus,
    ExplanationDepth,
    ExplanationResult,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackType,
    FinancialDiscrepancy,
    GuardrailSummary,
    HealthResponse,
    LearningMetrics,
    ModelLineage,
    ModelStatus,
    ModelVersion,
    OverallMetrics,
    PaginationParams,
    RejectRequest,
    ResolutionCandidateSummary,
    ResolveRequest,
    ResolveResponse,
    SafetyMetrics,
    SimilarCaseItem,
    SimilarCasesResponse,
    ThroughputMetrics,
)
from app.schemas.enums import ExceptionType, ResolutionType, RiskCategory


# ═══════════════════════════════════════════════════════════════════════════════
# Common Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class TestPaginationParams:
    def test_valid(self):
        p = PaginationParams(limit=10, offset=0)
        assert p.limit == 10

    def test_default(self):
        p = PaginationParams()
        assert p.limit == 50
        assert p.offset == 0

    def test_limit_too_high(self):
        with pytest.raises(ValidationError):
            PaginationParams(limit=501)

    def test_limit_zero(self):
        with pytest.raises(ValidationError):
            PaginationParams(limit=0)

    def test_negative_offset(self):
        with pytest.raises(ValidationError):
            PaginationParams(offset=-1)


class TestApiResponse:
    def test_success(self):
        r = ApiResponse(success=True, data={"key": "value"})
        assert r.success is True

    def test_failure(self):
        r = ApiResponse(success=False, error="Not found")
        assert r.success is False
        assert r.error == "Not found"


class TestHealthResponse:
    def test_default(self):
        h = HealthResponse()
        assert h.status == "ok"
        assert h.version == "1.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchCreateRequest:
    def test_valid(self):
        r = BatchCreateRequest(name="test-batch")
        assert r.name == "test-batch"

    def test_missing_name(self):
        with pytest.raises(ValidationError):
            BatchCreateRequest()

    def test_empty_name(self):
        with pytest.raises(ValidationError):
            BatchCreateRequest(name="")

    def test_name_too_long(self):
        with pytest.raises(ValidationError):
            BatchCreateRequest(name="x" * 201)


class TestBatchResponse:
    def test_valid(self):
        b = BatchResponse(batch_id="BATCH-001", name="Test")
        assert b.batch_id == "BATCH-001"
        assert b.status == BatchStatus.CREATED

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            BatchResponse(batch_id="B-1", name="X", status="INVALID")


class TestBatchSummaryResponse:
    def test_valid(self):
        s = BatchSummaryResponse(batch_id="B-1", status=BatchStatus.COMPLETED)
        assert s.total_exceptions == 0

    def test_empty_batch_id_allowed(self):
        # BatchSummaryResponse batch_id has no min_length constraint
        b = BatchSummaryResponse(batch_id="", status=BatchStatus.COMPLETED)
        assert b.batch_id == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Exception Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class TestExceptionListItem:
    def test_valid(self):
        e = ExceptionListItem(
            exception_id="EXC-001",
            case_id="CASE-001",
            merchant_id="M-001",
            payment_id="PAY-001",
            exception_type=ExceptionType.FEE_DIFFERENCE,
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=-5000,
            risk_category=RiskCategory.LOW,
        )
        assert e.exception_type == ExceptionType.FEE_DIFFERENCE

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError):
            ExceptionListItem(
                exception_id="EXC-001",
                case_id="CASE-001",
                merchant_id="M-001",
                payment_id="PAY-001",
                exception_type=ExceptionType.FEE_DIFFERENCE,
                expected_amount_paise=-100,
                actual_amount_paise=95000,
                difference_paise=-5000,
                risk_category=RiskCategory.LOW,
            )

    def test_invalid_exception_type(self):
        with pytest.raises(ValidationError):
            ExceptionListItem(
                exception_id="EXC-001",
                case_id="CASE-001",
                merchant_id="M-001",
                payment_id="PAY-001",
                exception_type="FAKE_TYPE",
                expected_amount_paise=100000,
                actual_amount_paise=95000,
                difference_paise=-5000,
                risk_category=RiskCategory.LOW,
            )

    def test_invalid_risk_category(self):
        with pytest.raises(ValidationError):
            ExceptionListItem(
                exception_id="EXC-001",
                case_id="CASE-001",
                merchant_id="M-001",
                payment_id="PAY-001",
                exception_type=ExceptionType.FEE_DIFFERENCE,
                expected_amount_paise=100000,
                actual_amount_paise=95000,
                difference_paise=-5000,
                risk_category="CRITICAL",
            )


class TestExceptionDetail:
    def test_valid(self):
        e = ExceptionDetail(
            exception_id="EXC-001",
            case_id="CASE-001",
            merchant_id="M-001",
            payment_id="PAY-001",
            exception_type=ExceptionType.REFUND_ADJUSTMENT,
            expected_amount_paise=200000,
            actual_amount_paise=180000,
            difference_paise=-20000,
            risk_category=RiskCategory.HIGH,
        )
        assert e.classification_confidence is None

    def test_confidence_bounds(self):
        e = ExceptionDetail(
            exception_id="EXC-001",
            case_id="CASE-001",
            merchant_id="M-001",
            payment_id="PAY-001",
            exception_type=ExceptionType.FEE_DIFFERENCE,
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=-5000,
            risk_category=RiskCategory.LOW,
            classification_confidence=0.95,
        )
        assert e.classification_confidence == 0.95

    def test_confidence_too_high(self):
        with pytest.raises(ValidationError):
            ExceptionDetail(
                exception_id="EXC-001",
                case_id="CASE-001",
                merchant_id="M-001",
                payment_id="PAY-001",
                exception_type=ExceptionType.FEE_DIFFERENCE,
                expected_amount_paise=100000,
                actual_amount_paise=95000,
                difference_paise=-5000,
                risk_category=RiskCategory.LOW,
                classification_confidence=1.5,
            )

    def test_confidence_negative(self):
        with pytest.raises(ValidationError):
            ExceptionDetail(
                exception_id="EXC-001",
                case_id="CASE-001",
                merchant_id="M-001",
                payment_id="PAY-001",
                exception_type=ExceptionType.FEE_DIFFERENCE,
                expected_amount_paise=100000,
                actual_amount_paise=95000,
                difference_paise=-5000,
                risk_category=RiskCategory.LOW,
                classification_confidence=-0.1,
            )


class TestResolveRequest:
    def test_valid(self):
        r = ResolveRequest(
            resolution_type=ResolutionType.FEE_ADJUSTMENT,
            adjustment_paise=5000,
        )
        assert r.adjustment_paise == 5000

    def test_large_adjustment_rejected(self):
        with pytest.raises(ValidationError):
            ResolveRequest(
                resolution_type=ResolutionType.FEE_ADJUSTMENT,
                adjustment_paise=20_000_000,  # exceeds 10 lakh limit
            )

    def test_negative_adjustment_valid(self):
        r = ResolveRequest(
            resolution_type=ResolutionType.REFUND_ADJUSTMENT,
            adjustment_paise=-50000,
        )
        assert r.adjustment_paise == -50000

    def test_invalid_resolution_type(self):
        with pytest.raises(ValidationError):
            ResolveRequest(resolution_type="FAKE")


class TestApproveRequest:
    def test_valid(self):
        a = ApproveRequest(approved_by="reviewer@example.com")
        assert a.approved_by == "reviewer@example.com"

    def test_empty_approved_by(self):
        with pytest.raises(ValidationError):
            ApproveRequest(approved_by="")

    def test_missing_approved_by(self):
        with pytest.raises(ValidationError):
            ApproveRequest()


class TestRejectRequest:
    def test_valid(self):
        r = RejectRequest(rejected_by="reviewer@example.com", reason="Incorrect")
        assert r.reason == "Incorrect"

    def test_empty_reason(self):
        with pytest.raises(ValidationError):
            RejectRequest(rejected_by="r@e.com", reason="")

    def test_missing_reason(self):
        with pytest.raises(ValidationError):
            RejectRequest(rejected_by="r@e.com")

    def test_reason_too_long(self):
        with pytest.raises(ValidationError):
            RejectRequest(rejected_by="r@e.com", reason="x" * 2001)


class TestEscalateRequest:
    def test_valid(self):
        e = EscalateRequest(reason="High risk")
        assert e.reason == "High risk"

    def test_empty_reason(self):
        with pytest.raises(ValidationError):
            EscalateRequest(reason="")

    def test_missing_reason(self):
        with pytest.raises(ValidationError):
            EscalateRequest()

    def test_reason_too_long(self):
        with pytest.raises(ValidationError):
            EscalateRequest(reason="x" * 2001)


# ═══════════════════════════════════════════════════════════════════════════════
# Intelligence Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalysisResult:
    def test_valid(self):
        a = AnalysisResult(
            exception_id="EXC-001",
            financial_discrepancy=FinancialDiscrepancy(
                expected_amount_paise=100000,
                actual_amount_paise=95000,
                difference_paise=-5000,
                exception_type=ExceptionType.FEE_DIFFERENCE,
            ),
            evidence=EvidenceSummary(),
            guardrail=GuardrailSummary(
                decision="HUMAN_REVIEW",
                confidence=0.7,
                risk_category=RiskCategory.MEDIUM,
            ),
        )
        assert a.fallback_used is True

    def test_invalid_decision(self):
        with pytest.raises(ValidationError):
            GuardrailSummary(
                decision="AUTO",
                confidence=1.5,  # out of range
                risk_category=RiskCategory.LOW,
            )


class TestExplanationResult:
    def test_valid(self):
        e = ExplanationResult(exception_id="EXC-001")
        assert e.fallback_used is True

    def test_with_evidence_coverage(self):
        e = ExplanationResult(
            exception_id="EXC-001",
            evidence_coverage=EvidenceCoverage.FULLY_EXPLAINED,
        )
        assert e.evidence_coverage == EvidenceCoverage.FULLY_EXPLAINED


class TestSimilarCaseItem:
    def test_valid(self):
        s = SimilarCaseItem(
            case_id="CASE-002",
            exception_type=ExceptionType.FEE_DIFFERENCE,
            similarity_score=0.92,
        )
        assert s.similarity_score == 0.92

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            SimilarCaseItem(
                case_id="CASE-002",
                exception_type=ExceptionType.FEE_DIFFERENCE,
                similarity_score=1.5,
            )

    def test_score_negative(self):
        with pytest.raises(ValidationError):
            SimilarCaseItem(
                case_id="CASE-002",
                exception_type=ExceptionType.FEE_DIFFERENCE,
                similarity_score=-0.1,
            )


class TestEvidenceRecord:
    def test_valid(self):
        e = EvidenceRecord(
            record_type="PAYMENT",
            record_id="PAY-001",
            amount_paise=100000,
        )
        assert e.record_type == "PAYMENT"

    def test_negative_amount(self):
        with pytest.raises(ValidationError):
            EvidenceRecord(
                record_type="PAYMENT",
                record_id="PAY-001",
                amount_paise=-100,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Learning / Feedback Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeedbackRequest:
    def test_approve(self):
        f = FeedbackRequest(
            feedback_type=FeedbackType.APPROVE,
            workflow_id="WF-001",
        )
        assert f.feedback_type == FeedbackType.APPROVE

    def test_reject(self):
        f = FeedbackRequest(
            feedback_type=FeedbackType.REJECT,
            workflow_id="WF-001",
            rejection_reason="Incorrect",
        )
        assert f.rejection_reason == "Incorrect"

    def test_correct(self):
        f = FeedbackRequest(
            feedback_type=FeedbackType.CORRECT,
            workflow_id="WF-001",
            corrected_resolution=ResolutionType.REFUND_ADJUSTMENT,
            correction_reason="Wrong amount",
        )
        assert f.corrected_resolution == ResolutionType.REFUND_ADJUSTMENT

    def test_escalate(self):
        f = FeedbackRequest(
            feedback_type=FeedbackType.ESCALATE,
            workflow_id="WF-001",
            escalation_reason="High value",
        )
        assert f.escalation_reason == "High value"

    def test_invalid_feedback_type(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(
                feedback_type="INVALID",
                workflow_id="WF-001",
            )

    def test_missing_workflow_id(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(feedback_type=FeedbackType.APPROVE)

    def test_empty_workflow_id(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(feedback_type=FeedbackType.APPROVE, workflow_id="")

    def test_workflow_id_too_long(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(
                feedback_type=FeedbackType.APPROVE,
                workflow_id="W" * 101,
            )


class TestLearningMetrics:
    def test_valid(self):
        m = LearningMetrics()
        assert m.automation_rate == 0.0

    def test_rate_out_of_range(self):
        with pytest.raises(ValidationError):
            LearningMetrics(automation_rate=1.5)

    def test_rate_negative(self):
        with pytest.raises(ValidationError):
            LearningMetrics(precision=-0.1)


class TestDatasetInfo:
    def test_valid(self):
        d = DatasetInfo(total_examples=100)
        assert d.total_examples == 100

    def test_negative_examples(self):
        with pytest.raises(ValidationError):
            DatasetInfo(total_examples=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class TestOverallMetrics:
    def test_valid(self):
        m = OverallMetrics()
        assert m.total_exceptions == 0

    def test_rate_out_of_range(self):
        with pytest.raises(ValidationError):
            OverallMetrics(automation_rate=2.0)


class TestSafetyMetrics:
    def test_valid(self):
        s = SafetyMetrics()
        assert s.guardrail_pass_rate == 1.0

    def test_rate_out_of_range(self):
        with pytest.raises(ValidationError):
            SafetyMetrics(false_automation_rate=1.5)

    def test_negative_exposure(self):
        with pytest.raises(ValidationError):
            SafetyMetrics(unsafe_exposure_paise=-1)


class TestThroughputMetrics:
    def test_valid(self):
        t = ThroughputMetrics()
        assert t.exceptions_per_hour == 0.0

    def test_negative_throughput(self):
        with pytest.raises(ValidationError):
            ThroughputMetrics(exceptions_per_hour=-1.0)


class TestBatchMetrics:
    def test_valid(self):
        b = BatchMetrics(batch_id="B-1")
        assert b.batch_id == "B-1"

    def test_rate_out_of_range(self):
        with pytest.raises(ValidationError):
            BatchMetrics(batch_id="B-1", precision=1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Model Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelVersion:
    def test_valid(self):
        m = ModelVersion(
            model_id="M-1",
            model_name="classifier",
            model_version="v1",
        )
        assert m.status == ModelStatus.CANDIDATE

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            ModelVersion(
                model_id="M-1",
                model_name="c",
                model_version="v1",
                status="INVALID",
            )

    def test_precision_bounds(self):
        m = ModelVersion(
            model_id="M-1",
            model_name="c",
            model_version="v1",
            precision=0.95,
        )
        assert m.precision == 0.95

    def test_precision_out_of_range(self):
        with pytest.raises(ValidationError):
            ModelVersion(
                model_id="M-1",
                model_name="c",
                model_version="v1",
                precision=2.0,
            )


class TestModelLineage:
    def test_valid(self):
        l = ModelLineage(model_id="M-1", model_version="v1")
        assert l.artifacts == []

    def test_with_config(self):
        l = ModelLineage(
            model_id="M-1",
            model_version="v1",
            training_config={"n_estimators": 100},
            metrics={"precision": 0.92},
        )
        assert l.training_config["n_estimators"] == 100


# ═══════════════════════════════════════════════════════════════════════════════
# Enum Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnumValidation:
    def test_all_exception_types_accepted(self):
        for et in ExceptionType:
            e = ExceptionListItem(
                exception_id="EXC-001",
                case_id="CASE-001",
                merchant_id="M-001",
                payment_id="PAY-001",
                exception_type=et,
                expected_amount_paise=100000,
                actual_amount_paise=95000,
                difference_paise=-5000,
                risk_category=RiskCategory.LOW,
            )
            assert e.exception_type == et

    def test_all_resolution_types_accepted(self):
        for rt in ResolutionType:
            r = ResolveRequest(resolution_type=rt, adjustment_paise=0)
            assert r.resolution_type == rt

    def test_all_risk_categories_accepted(self):
        for rc in RiskCategory:
            g = GuardrailSummary(
                decision="AUTO",
                confidence=0.9,
                risk_category=rc,
            )
            assert g.risk_category == rc

    def test_all_feedback_types_accepted(self):
        for ft in FeedbackType:
            f = FeedbackRequest(feedback_type=ft, workflow_id="WF-001")
            assert f.feedback_type == ft

    def test_all_batch_statuses_accepted(self):
        for bs in BatchStatus:
            b = BatchResponse(batch_id="B-1", name="X", status=bs)
            assert b.status == bs

    def test_all_exception_statuses_accepted(self):
        for es in ExceptionStatus:
            e = ExceptionListItem(
                exception_id="EXC-001",
                case_id="CASE-001",
                merchant_id="M-001",
                payment_id="PAY-001",
                exception_type=ExceptionType.FEE_DIFFERENCE,
                expected_amount_paise=100000,
                actual_amount_paise=95000,
                difference_paise=-5000,
                risk_category=RiskCategory.LOW,
                status=es,
            )
            assert e.status == es

    def test_all_model_statuses_accepted(self):
        for ms in ModelStatus:
            m = ModelVersion(
                model_id="M-1",
                model_name="c",
                model_version="v1",
                status=ms,
            )
            assert m.status == ms

    def test_all_evidence_coverages_accepted(self):
        for ec in EvidenceCoverage:
            e = ExplanationResult(
                exception_id="EXC-001",
                evidence_coverage=ec,
            )
            assert e.evidence_coverage == ec


# ═══════════════════════════════════════════════════════════════════════════════
# Oversized / Edge Case Values
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_very_large_paise_amount(self):
        """10 lakh paise = ₹100,000 — should be accepted."""
        r = ResolveRequest(
            resolution_type=ResolutionType.FEE_ADJUSTMENT,
            adjustment_paise=10_000_000,
        )
        assert r.adjustment_paise == 10_000_000

    def test_very_large_paise_amount_rejected(self):
        """Over ₹100,000 — should be rejected."""
        with pytest.raises(ValidationError):
            ResolveRequest(
                resolution_type=ResolutionType.FEE_ADJUSTMENT,
                adjustment_paise=10_000_001,
            )

    def test_zero_amount_valid(self):
        r = ResolveRequest(
            resolution_type=ResolutionType.NO_ACTION,
            adjustment_paise=0,
        )
        assert r.adjustment_paise == 0

    def test_confidence_boundary_zero(self):
        e = ExceptionDetail(
            exception_id="EXC-001",
            case_id="CASE-001",
            merchant_id="M-001",
            payment_id="PAY-001",
            exception_type=ExceptionType.FEE_DIFFERENCE,
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=-5000,
            risk_category=RiskCategory.LOW,
            classification_confidence=0.0,
        )
        assert e.classification_confidence == 0.0

    def test_confidence_boundary_one(self):
        e = ExceptionDetail(
            exception_id="EXC-001",
            case_id="CASE-001",
            merchant_id="M-001",
            payment_id="PAY-001",
            exception_type=ExceptionType.FEE_DIFFERENCE,
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=-5000,
            risk_category=RiskCategory.LOW,
            classification_confidence=1.0,
        )
        assert e.classification_confidence == 1.0

    def test_similarity_boundary_zero(self):
        s = SimilarCaseItem(
            case_id="C-1",
            exception_type=ExceptionType.FEE_DIFFERENCE,
            similarity_score=0.0,
        )
        assert s.similarity_score == 0.0

    def test_similarity_boundary_one(self):
        s = SimilarCaseItem(
            case_id="C-1",
            exception_type=ExceptionType.FEE_DIFFERENCE,
            similarity_score=1.0,
        )
        assert s.similarity_score == 1.0

    def test_max_pagination_limit(self):
        p = PaginationParams(limit=500, offset=0)
        assert p.limit == 500

    def test_max_feedback_workflow_id(self):
        f = FeedbackRequest(
            feedback_type=FeedbackType.APPROVE,
            workflow_id="W" * 100,
        )
        assert len(f.workflow_id) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Serialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestSerialization:
    def test_resolve_request_serializes(self):
        r = ResolveRequest(
            resolution_type=ResolutionType.FEE_ADJUSTMENT,
            adjustment_paise=5000,
            reason="Test",
        )
        d = r.model_dump()
        assert d["resolution_type"] == "FEE_ADJUSTMENT"
        assert d["adjustment_paise"] == 5000

    def test_feedback_request_excludes_none(self):
        f = FeedbackRequest(
            feedback_type=FeedbackType.APPROVE,
            workflow_id="WF-001",
        )
        d = f.model_dump(exclude_none=True)
        assert "correction_reason" not in d
        assert "rejection_reason" not in d

    def test_exception_list_item_json(self):
        e = ExceptionListItem(
            exception_id="EXC-001",
            case_id="CASE-001",
            merchant_id="M-001",
            payment_id="PAY-001",
            exception_type=ExceptionType.FEE_DIFFERENCE,
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=-5000,
            risk_category=RiskCategory.LOW,
        )
        j = e.model_dump_json()
        assert "EXC-001" in j
        assert "FEE_DIFFERENCE" in j
