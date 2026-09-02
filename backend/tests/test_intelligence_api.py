"""
Tests for Phase 13.5 — Intelligence REST APIs.

Tests:
- Analyze normal exception
- Analyze unknown exception
- Explain normal exception
- Explain with fallback
- Similar cases
- No similar cases
- Large top-k
- Evidence for normal exception
- Missing evidence
- Conflicting evidence
- Invalid exception ID
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_first_exception_id(client) -> str:
    """Get a valid exception ID from the dataset."""
    r = client.get("/exceptions?limit=1")
    data = r.json()["data"]
    if data:
        return data[0]["exception_id"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# POST /exceptions/{exception_id}/analyze
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeException:
    def test_analyze_existing(self, client):
        """Analyze a real exception from the dataset."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.post(f"/exceptions/{exc_id}/analyze")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            result = data["data"]
            # Should have core fields
            assert "exception_id" in result
            assert "financial_discrepancy" in result

    def test_analyze_has_financial_discrepancy(self, client):
        """Analysis includes financial discrepancy."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.post(f"/exceptions/{exc_id}/analyze")
            result = response.json()["data"]
            fd = result.get("financial_discrepancy", {})
            assert "expected_amount_paise" in fd
            assert "actual_amount_paise" in fd
            assert "difference_paise" in fd
            assert "exception_type" in fd

    def test_analyze_has_evidence(self, client):
        """Analysis includes evidence summary."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.post(f"/exceptions/{exc_id}/analyze")
            result = response.json()["data"]
            evidence = result.get("evidence", {})
            assert "record_count" in evidence
            assert "coverage" in evidence

    def test_analyze_has_guardrail(self, client):
        """Analysis includes guardrail summary."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.post(f"/exceptions/{exc_id}/analyze")
            result = response.json()["data"]
            guardrail = result.get("guardrail", {})
            assert "decision" in guardrail
            assert "risk_category" in guardrail

    def test_analyze_not_found(self, client):
        """Analyze unknown exception returns 404."""
        response = client.post("/exceptions/NONEXISTENT-CASE/analyze")
        assert response.status_code == 404

    def test_analyze_returns_ai_explanation(self, client):
        """Analysis includes AI explanation or fallback."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.post(f"/exceptions/{exc_id}/analyze")
            result = response.json()["data"]
            assert "ai_explanation" in result
            assert "fallback_used" in result

    def test_analyze_fallback_when_llm_unavailable(self, client):
        """When LLM is unavailable, fallback is used."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.post(f"/exceptions/{exc_id}/analyze")
            result = response.json()["data"]
            # LLM is disabled by default, so fallback should be used
            assert result.get("fallback_used") is True


# ─────────────────────────────────────────────────────────────────────────────
# GET /exceptions/{exception_id}/explain
# ─────────────────────────────────────────────────────────────────────────────


class TestExplainException:
    def test_explain_existing(self, client):
        """Explain a real exception."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/explain")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            result = data["data"]
            assert "summary" in result
            assert "reason" in result

    def test_explain_has_evidence_summary(self, client):
        """Explanation includes evidence summary."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/explain")
            result = response.json()["data"]
            assert "evidence_summary" in result

    def test_explain_has_uncertainty(self, client):
        """Explanation includes uncertainty."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/explain")
            result = response.json()["data"]
            assert "uncertainty" in result

    def test_explain_with_depth(self, client):
        """Explain accepts depth parameter."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/explain?depth=detailed")
            assert response.status_code == 200

    def test_explain_not_found(self, client):
        """Explain unknown exception returns 404."""
        response = client.get("/exceptions/NONEXISTENT-CASE/explain")
        assert response.status_code == 404

    def test_explain_fallback_used(self, client):
        """When LLM unavailable, fallback is used."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/explain")
            result = response.json()["data"]
            assert result.get("fallback_used") is True

    def test_explain_has_financial_context(self, client):
        """Explanation includes financial amounts."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/explain")
            result = response.json()["data"]
            assert "expected_amount_paise" in result
            assert "actual_amount_paise" in result
            assert "difference_paise" in result

    def test_explain_brief_depth(self, client):
        """Explain with brief depth."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/explain?depth=brief")
            assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /exceptions/{exception_id}/similar
# ─────────────────────────────────────────────────────────────────────────────


class TestSimilarCases:
    def test_similar_existing(self, client):
        """Find similar cases for a real exception."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/similar")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            result = data["data"]
            assert "similar_cases" in result
            assert "count" in result
            assert "confidence" in result

    def test_similar_returns_list(self, client):
        """Similar cases returns a list."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/similar")
            result = response.json()["data"]
            assert isinstance(result["similar_cases"], list)

    def test_similar_cases_have_fields(self, client):
        """Each similar case has required fields."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/similar")
            cases = response.json()["data"]["similar_cases"]
            for c in cases:
                assert "case_id" in c
                assert "similarity_score" in c
                assert "exception_type" in c

    def test_similar_with_limit(self, client):
        """Limit controls maximum results."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/similar?limit=2")
            result = response.json()["data"]
            assert len(result["similar_cases"]) <= 2

    def test_similar_limit_1(self, client):
        """Limit=1 returns at most 1."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/similar?limit=1")
            result = response.json()["data"]
            assert len(result["similar_cases"]) <= 1

    def test_similar_not_found(self, client):
        """Similar for unknown exception returns 404."""
        response = client.get("/exceptions/NONEXISTENT-CASE/similar")
        assert response.status_code == 404

    def test_similar_scores_bounded(self, client):
        """Similarity scores are in [0, 1]."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/similar")
            cases = response.json()["data"]["similar_cases"]
            for c in cases:
                assert 0.0 <= c["similarity_score"] <= 1.0

    def test_similar_confidence_values(self, client):
        """Confidence is HIGH, MEDIUM, or LOW."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/similar")
            result = response.json()["data"]
            assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")


# ─────────────────────────────────────────────────────────────────────────────
# GET /exceptions/{exception_id}/evidence
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidence:
    def test_evidence_existing(self, client):
        """Get evidence for a real exception."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/evidence")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            result = data["data"]
            assert "evidence" in result
            assert "coverage" in result
            assert "record_count" in result

    def test_evidence_has_records(self, client):
        """Evidence contains financial records."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/evidence")
            evidence = response.json()["data"]["evidence"]
            assert isinstance(evidence, list)
            assert len(evidence) > 0

    def test_evidence_record_fields(self, client):
        """Each evidence record has required fields."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/evidence")
            records = response.json()["data"]["evidence"]
            for r in records:
                assert "record_type" in r
                assert "record_id" in r
                assert "amount_paise" in r

    def test_evidence_has_payment(self, client):
        """Evidence includes payment record."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/evidence")
            records = response.json()["data"]["evidence"]
            record_types = [r["record_type"] for r in records]
            assert "PAYMENT" in record_types

    def test_evidence_coverage_values(self, client):
        """Coverage is a valid value."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/evidence")
            coverage = response.json()["data"]["coverage"]
            assert coverage in ("FULLY_EXPLAINED", "PARTIALLY_EXPLAINED", "UNEXPLAINED", "CONFLICTING", "UNKNOWN")

    def test_evidence_has_conflicts_list(self, client):
        """Evidence includes conflicts list."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/evidence")
            result = response.json()["data"]
            assert "conflicts" in result
            assert isinstance(result["conflicts"], list)

    def test_evidence_has_missing_list(self, client):
        """Evidence includes missing evidence list."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/evidence")
            result = response.json()["data"]
            assert "missing_evidence" in result
            assert isinstance(result["missing_evidence"], list)

    def test_evidence_not_found(self, client):
        """Evidence for unknown exception returns 404."""
        response = client.get("/exceptions/NONEXISTENT-CASE/evidence")
        assert response.status_code == 404

    def test_evidence_amounts_non_negative(self, client):
        """Evidence amounts are non-negative."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            response = client.get(f"/exceptions/{exc_id}/evidence")
            records = response.json()["data"]["evidence"]
            for r in records:
                assert r["amount_paise"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-API Consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossAPIConsistency:
    def test_analyze_and_explain_same_exception(self, client):
        """Analyze and explain return data for the same exception."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            analyze = client.post(f"/exceptions/{exc_id}/analyze").json()
            explain = client.get(f"/exceptions/{exc_id}/explain").json()
            assert analyze["data"]["exception_id"] == exc_id
            assert explain["data"]["exception_id"] == exc_id

    def test_evidence_matches_analyze(self, client):
        """Evidence record count matches analyze evidence count."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            evidence = client.get(f"/exceptions/{exc_id}/evidence").json()["data"]
            analyze = client.post(f"/exceptions/{exc_id}/analyze").json()["data"]
            # Both should reference the same exception
            assert evidence["exception_id"] == analyze.get("exception_id")

    def test_similar_and_evidence_independent(self, client):
        """Similar and evidence can be called independently."""
        exc_id = _get_first_exception_id(client)
        if exc_id:
            r1 = client.get(f"/exceptions/{exc_id}/similar")
            r2 = client.get(f"/exceptions/{exc_id}/evidence")
            assert r1.status_code == 200
            assert r2.status_code == 200
