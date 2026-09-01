"""
Tests for Razorpay CloseLoop Phase 11C — MCP Input Validation + Security.

Verifies that all MCP inputs are validated, bounded, and safe.
"""

import pytest

from mcp.input_validation import (
    BLOCKED_PARAMETER_NAMES,
    DANGEROUS_PATTERNS,
    MAX_SEARCH_LIMIT,
    MAX_STRING_LENGTH,
    MAX_TOP_K,
    ValidationResult,
    validate_amount,
    validate_id,
    validate_limit,
    validate_no_injection,
    validate_output,
    validate_parameter_names,
    validate_record_type,
    validate_string_safety,
    validate_tool_parameters,
)
from mcp.tools.readonly import TOOL_DEFINITIONS, create_handlers
from mcp.adapters.financial_data import FinancialDataAdapter
from mcp.server import MCPServer
from mcp.schemas import MCPToolRequest, MCPToolStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_data_dir(tmp_path):
    batch_dir = tmp_path / "batch_001" / "generated"
    batch_dir.mkdir(parents=True)

    import json
    payments = [
        {"payment_id": "PAY-001", "merchant_id": "MER-001", "amount": 10000, "currency": "INR"},
        {"payment_id": "PAY-002", "merchant_id": "MER-001", "amount": 20000, "currency": "INR"},
    ]
    settlements = [
        {"settlement_id": "SET-001", "payment_id": "PAY-001", "merchant_id": "MER-001", "amount": 9800},
    ]
    for name, data in [
        ("payments.json", payments), ("settlements.json", settlements),
        ("refunds.json", []), ("fees.json", []), ("adjustments.json", []),
        ("cases.json", [
            {"case_id": "CASE-001", "payment_id": "PAY-001", "merchant_id": "MER-001", "scenario": "FEE_DIFFERENCE", "difference": -200, "risk_category": "low"},
        ]),
        ("merchants.json", [{"merchant_id": "MER-001", "name": "Test"}]),
    ]:
        (batch_dir / name).write_text(json.dumps(data))

    return str(tmp_path)


@pytest.fixture
def adapter(sample_data_dir) -> FinancialDataAdapter:
    a = FinancialDataAdapter(data_dir=sample_data_dir)
    a.load_batch("batch_001")
    return a


@pytest.fixture
def server_with_tools(adapter) -> MCPServer:
    server = MCPServer()
    handlers = create_handlers(adapter)
    for defn in TOOL_DEFINITIONS:
        if defn.name in handlers:
            server.register_tool(defn, handlers[defn.name])
    return server


# ─────────────────────────────────────────────────────────────────────────────
# ValidationResult
# ─────────────────────────────────────────────────────────────────────────────


class TestValidationResult:
    def test_valid(self):
        r = ValidationResult()
        assert r.is_valid is True
        assert r.errors == []
        assert r.error_message is None

    def test_reject(self):
        r = ValidationResult()
        r.reject("bad input")
        assert r.is_valid is False
        assert len(r.errors) == 1
        assert "bad input" in r.error_message

    def test_multiple_rejects(self):
        r = ValidationResult()
        r.reject("error 1")
        r.reject("error 2")
        assert len(r.errors) == 2


# ─────────────────────────────────────────────────────────────────────────────
# ID Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateID:
    def test_valid_id(self):
        assert validate_id("PAY-001").is_valid
        assert validate_id("SET-001").is_valid
        assert validate_id("CASE-001").is_valid
        assert validate_id("MER-001").is_valid
        assert validate_id("model-abc123").is_valid

    def test_valid_underscore(self):
        assert validate_id("PAY_001").is_valid

    def test_empty_id(self):
        assert not validate_id("").is_valid

    def test_none_id(self):
        assert not validate_id(None).is_valid

    def test_too_long(self):
        assert not validate_id("X" * 52).is_valid

    def test_exact_max_length(self):
        assert validate_id("X" * 51).is_valid

    def test_special_characters(self):
        assert not validate_id("PAY; DROP").is_valid
        assert not validate_id("PAY' OR 1=1").is_valid
        assert not validate_id("PAY`inject`").is_valid

    def test_spaces(self):
        assert not validate_id("PAY 001").is_valid

    def test_starts_with_number(self):
        assert not validate_id("1PAY").is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Limit Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateLimit:
    def test_valid_limit(self):
        assert validate_limit(50).is_valid
        assert validate_limit(0).is_valid
        assert validate_limit(1).is_valid

    def test_none_limit(self):
        assert validate_limit(None).is_valid  # Optional

    def test_string_number(self):
        assert validate_limit("10").is_valid

    def test_negative(self):
        assert not validate_limit(-1).is_valid

    def test_exceeds_max(self):
        assert not validate_limit(MAX_SEARCH_LIMIT + 1).is_valid

    def test_at_max(self):
        assert validate_limit(MAX_SEARCH_LIMIT).is_valid

    def test_non_numeric(self):
        assert not validate_limit("abc").is_valid

    def test_float_string(self):
        assert not validate_limit("10.5").is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Record Type Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateRecordType:
    def test_valid_types(self):
        for rt in ["payment", "settlement", "refund", "fee", "adjustment", "case"]:
            assert validate_record_type(rt).is_valid

    def test_case_insensitive(self):
        assert validate_record_type("Payment").is_valid
        assert validate_record_type("SETTLEMENT").is_valid

    def test_invalid_type(self):
        assert not validate_record_type("unknown").is_valid
        assert not validate_record_type("transaction").is_valid

    def test_none(self):
        assert validate_record_type(None).is_valid  # Optional


# ─────────────────────────────────────────────────────────────────────────────
# Amount Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateAmount:
    def test_valid_amount(self):
        assert validate_amount(10000).is_valid
        assert validate_amount(0).is_valid

    def test_string_amount(self):
        assert validate_amount("10000").is_valid

    def test_negative_disallowed(self):
        assert not validate_amount(-100).is_valid

    def test_negative_allowed(self):
        assert validate_amount(-100, allow_negative=True).is_valid

    def test_none(self):
        assert validate_amount(None).is_valid  # Optional

    def test_non_numeric(self):
        assert not validate_amount("abc").is_valid


# ─────────────────────────────────────────────────────────────────────────────
# String Safety Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestStringSafety:
    def test_safe_string(self):
        assert validate_string_safety("hello world").is_valid

    def test_too_long(self):
        assert not validate_string_safety("x" * (MAX_STRING_LENGTH + 1)).is_valid

    def test_exact_max(self):
        assert validate_string_safety("x" * MAX_STRING_LENGTH).is_valid

    def test_none(self):
        assert validate_string_safety(None).is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Name Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestParameterNames:
    def test_valid_names(self):
        assert validate_parameter_names({"payment_id": "PAY-001"}, set()).is_valid

    def test_blocked_sql(self):
        assert not validate_parameter_names({"sql": "SELECT *"}, set()).is_valid

    def test_blocked_exec(self):
        assert not validate_parameter_names({"exec": "os.system('rm -rf /')"}, set()).is_valid

    def test_blocked_eval(self):
        assert not validate_parameter_names({"eval": "1+1"}, set()).is_valid

    def test_blocked_proto(self):
        assert not validate_parameter_names({"__proto__": {}}, set()).is_valid

    def test_blocked_query(self):
        assert not validate_parameter_names({"query": "SHOW TABLES"}, set()).is_valid

    def test_blocked_shell(self):
        assert not validate_parameter_names({"shell": "bash"}, set()).is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Injection Detection
# ─────────────────────────────────────────────────────────────────────────────


class TestInjectionDetection:
    def test_sql_drop(self):
        assert not validate_no_injection("'; DROP TABLE payments; --").is_valid

    def test_sql_select(self):
        assert not validate_no_injection("SELECT * FROM users").is_valid

    def test_sql_union(self):
        assert not validate_no_injection("UNION SELECT password FROM users").is_valid

    def test_sql_comment(self):
        assert not validate_no_injection("PAY-001 -- comment").is_valid

    def test_script_tag(self):
        assert not validate_no_injection("<script>alert('xss')</script>").is_valid

    def test_system_command(self):
        assert not validate_no_injection("SHELL rm -rf /").is_valid

    def test_eval_command(self):
        assert not validate_no_injection("EVAL malicious_code").is_valid

    def test_import_command(self):
        assert not validate_no_injection("IMPORT os; os.system('ls')").is_valid

    def test_or_injection(self):
        assert not validate_no_injection("' OR '1'='1").is_valid

    def test_safe_value(self):
        assert validate_no_injection("PAY-001").is_valid
        assert validate_no_injection("FEE_DIFFERENCE").is_valid
        assert validate_no_injection("10000").is_valid

    def test_dict_injection(self):
        assert not validate_no_injection({"key": "DROP TABLE x"}).is_valid

    def test_list_injection(self):
        assert not validate_no_injection(["safe", "DROP TABLE x"]).is_valid

    def test_nested_dict_injection(self):
        assert not validate_no_injection({"a": {"b": "SELECT * FROM x"}}).is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Composite Parameter Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestCompositeValidation:
    def test_valid_params(self):
        val = validate_tool_parameters(
            {"payment_id": "PAY-001"},
            required_params={"payment_id"},
            id_params={"payment_id"},
        )
        assert val.is_valid

    def test_missing_required(self):
        val = validate_tool_parameters(
            {},
            required_params={"payment_id"},
        )
        assert not val.is_valid
        assert "payment_id" in val.error_message

    def test_invalid_id(self):
        val = validate_tool_parameters(
            {"payment_id": "PAY; DROP TABLE"},
            required_params={"payment_id"},
            id_params={"payment_id"},
        )
        assert not val.is_valid

    def test_invalid_record_type(self):
        val = validate_tool_parameters(
            {"record_type": "unknown_type"},
            required_params=set(),
            record_type_params={"record_type"},
        )
        assert not val.is_valid

    def test_invalid_limit(self):
        val = validate_tool_parameters(
            {"limit": 99999},
            required_params=set(),
            limit_params={"limit"},
        )
        assert not val.is_valid

    def test_injection_in_value(self):
        val = validate_tool_parameters(
            {"payment_id": "PAY-001'; DROP TABLE payments; --"},
            required_params={"payment_id"},
            id_params={"payment_id"},
        )
        assert not val.is_valid

    def test_all_valid(self):
        val = validate_tool_parameters(
            {
                "payment_id": "PAY-001",
                "record_type": "payment",
                "limit": 10,
            },
            required_params=set(),
            id_params={"payment_id"},
            record_type_params={"record_type"},
            limit_params={"limit"},
        )
        assert val.is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Output Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestOutputValidation:
    def test_valid_output(self):
        assert validate_output({"count": 5, "records": []}).is_valid

    def test_empty_output(self):
        assert validate_output({}).is_valid

    def test_sensitive_password(self):
        assert not validate_output({"password": "secret123"}).is_valid

    def test_sensitive_token(self):
        assert not validate_output({"token": "abc123"}).is_valid

    def test_sensitive_api_key(self):
        assert not validate_output({"api_key": "key123"}).is_valid

    def test_nested_sensitive(self):
        assert not validate_output({"config": {"password": "x"}}).is_valid

    def test_stack_trace(self):
        assert not validate_output({"error": "Traceback (most recent call last)"}).is_valid

    def test_file_traceback(self):
        assert not validate_output({"log": "File \"/app/main.py\", line 10"}).is_valid

    def test_safe_nested(self):
        assert validate_output({"a": {"b": "safe"}}).is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Tool-Level Validation (via server)
# ─────────────────────────────────────────────────────────────────────────────


class TestToolLevelValidation:
    def test_sql_injection_in_payment_id(self, server_with_tools):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001'; DROP TABLE payments; --"},
        ))
        # Should be rejected by validation
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result

    def test_sql_injection_in_search(self, server_with_tools):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"payment_id": "'; SELECT * FROM users; --"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result

    def test_excessive_limit(self, server_with_tools):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"limit": 99999},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result

    def test_invalid_record_type(self, server_with_tools):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"record_type": "unknown_type"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result

    def test_blocked_parameter(self, server_with_tools):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"sql": "SELECT * FROM payments"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result

    def test_injection_in_similar_exception(self, server_with_tools):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_similar_exception",
            parameters={"exception_id": "CASE-001; DROP TABLE cases; --"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result

    def test_excessive_top_k(self, server_with_tools):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_similar_exception",
            parameters={"exception_id": "CASE-001", "top_k": 99999},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result

    def test_valid_request_passes(self, server_with_tools):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result.get("found") is True

    def test_valid_search_passes(self, server_with_tools):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"merchant_id": "MER-001", "limit": 10},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result.get("count", 0) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Constants Verification
# ─────────────────────────────────────────────────────────────────────────────


class TestConstants:
    def test_max_search_limit(self):
        assert MAX_SEARCH_LIMIT == 100

    def test_max_top_k(self):
        assert MAX_TOP_K == 50

    def test_max_string_length(self):
        assert MAX_STRING_LENGTH == 200

    def test_blocked_params_exist(self):
        assert "sql" in BLOCKED_PARAMETER_NAMES
        assert "exec" in BLOCKED_PARAMETER_NAMES
        assert "eval" in BLOCKED_PARAMETER_NAMES
        assert "query" in BLOCKED_PARAMETER_NAMES
        assert "__proto__" in BLOCKED_PARAMETER_NAMES

    def test_dangerous_patterns_exist(self):
        assert len(DANGEROUS_PATTERNS) >= 5


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_unicode_injection(self):
        val = validate_no_injection("PAY\u2014001")  # Em dash
        assert val.is_valid  # Unicode in ID is fine (not SQL)

    def test_empty_dict_params(self):
        val = validate_tool_parameters({}, required_params=set())
        assert val.is_valid

    def test_none_values(self):
        val = validate_tool_parameters(
            {"payment_id": None},
            required_params={"payment_id"},
        )
        assert not val.is_valid

    def test_numeric_id(self):
        assert not validate_id("12345").is_valid  # Must start with letter

    def test_very_long_error_message(self):
        val = ValidationResult()
        for i in range(100):
            val.reject(f"error_{i}")
        assert not val.is_valid
        assert len(val.errors) == 100

    def test_zero_limit(self):
        assert validate_limit(0).is_valid  # Empty result is valid

    def test_string_zero_limit(self):
        assert validate_limit("0").is_valid
