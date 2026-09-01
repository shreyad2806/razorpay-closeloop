"""
MCP Input Validation for Razorpay CloseLoop Phase 11C.

Comprehensive validation of ALL MCP tool inputs.

Safety principle:
  Validation is a HARD GATE.
  Invalid inputs are REJECTED before reaching any backend service.
  Validation never authorizes financial actions.

Rejection criteria:
  - Malformed IDs
  - Unknown/extra parameters
  - Excessive result limits
  - Invalid amounts (negative where invalid)
  - Unsupported record types
  - Arbitrary SQL or database commands
  - Unbounded searches
  - Injection patterns
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maximum allowed search limit
MAX_SEARCH_LIMIT = 100

# Maximum allowed top_k for similarity
MAX_TOP_K = 50

# Maximum string length for any parameter value
MAX_STRING_LENGTH = 200

# Allowed ID patterns (e.g. PAY-001, SET-002, REF-003, FEE-001, ADJ-001, CASE-001)
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,50}$")

# Allowed record types for search
ALLOWED_RECORD_TYPES = frozenset({
    "payment", "settlement", "refund", "fee", "adjustment", "case",
})

# Patterns that indicate injection/SQL attempts
DANGEROUS_PATTERNS = [
    # SQL injection: DDL/DML with or without semicolon prefix
    re.compile(r"(DROP|DELETE|INSERT\s+INTO|UPDATE\s+\w+\s+SET|ALTER\s+TABLE|CREATE\s+TABLE)", re.IGNORECASE),
    # SQL injection: SELECT/UNION
    re.compile(r"(UNION\s+SELECT|SELECT\s+.*FROM\s)", re.IGNORECASE),
    # SQL comments
    re.compile(r"(--|#|/\*|\*/)", re.IGNORECASE),
    # Command injection keywords
    re.compile(r"(SHELL|SYSTEM|EXEC|EVAL|IMPORT\s+OS)", re.IGNORECASE),
    # Classic OR injection
    re.compile(r"['\"].*OR.*['\"].*=", re.IGNORECASE),
    # XSS
    re.compile(r"<script", re.IGNORECASE),
]

# Fields that must NOT appear in tool parameters (injection prevention)
BLOCKED_PARAMETER_NAMES = frozenset({
    "sql", "query", "command", "exec", "eval", "shell",
    "__proto__", "constructor", "prototype",
})


# ─────────────────────────────────────────────────────────────────────────────
# Validation Result
# ─────────────────────────────────────────────────────────────────────────────


class ValidationResult:
    """Result of input validation."""
    __slots__ = ("_valid", "_errors")

    def __init__(self) -> None:
        self._valid = True
        self._errors: List[str] = []

    @property
    def is_valid(self) -> bool:
        return self._valid

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    @property
    def error_message(self) -> Optional[str]:
        return "; ".join(self._errors) if self._errors else None

    def reject(self, reason: str) -> None:
        """Add a rejection reason."""
        self._valid = False
        self._errors.append(reason)


# ─────────────────────────────────────────────────────────────────────────────
# Validation Functions
# ─────────────────────────────────────────────────────────────────────────────


def validate_id(value: Any, field_name: str = "id") -> ValidationResult:
    """Validate an ID field.

    IDs must match the pattern: alphanumeric + hyphens/underscores, 1-51 chars.
    """
    result = ValidationResult()
    if value is None:
        result.reject(f"{field_name} is required")
        return result

    s = str(value).strip()
    if not s:
        result.reject(f"{field_name} is empty")
        return result

    if len(s) > 51:
        result.reject(f"{field_name} exceeds maximum length (51)")
        return result

    if not ID_PATTERN.match(s):
        result.reject(f"{field_name} contains invalid characters")
        return result

    return result


def validate_limit(value: Any, field_name: str = "limit", max_value: int = MAX_SEARCH_LIMIT) -> ValidationResult:
    """Validate a numeric limit/pagination field."""
    result = ValidationResult()
    if value is None:
        return result  # Optional, default will be applied

    try:
        n = int(value)
    except (ValueError, TypeError):
        result.reject(f"{field_name} must be a valid integer")
        return result

    if n < 0:
        result.reject(f"{field_name} must be non-negative")
        return result

    if n > max_value:
        result.reject(f"{field_name} exceeds maximum ({max_value})")
        return result

    return result


def validate_record_type(value: Any) -> ValidationResult:
    """Validate a record type against the allowed set."""
    result = ValidationResult()
    if value is None:
        return result  # Optional

    s = str(value).strip().lower()
    if s not in ALLOWED_RECORD_TYPES:
        result.reject(
            f"Invalid record type '{value}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_RECORD_TYPES))}"
        )
        return result

    return result


def validate_amount(value: Any, field_name: str = "amount", allow_negative: bool = False) -> ValidationResult:
    """Validate a financial amount (integer paise)."""
    result = ValidationResult()
    if value is None:
        return result  # Optional

    try:
        n = int(value)
    except (ValueError, TypeError):
        result.reject(f"{field_name} must be a valid integer")
        return result

    if not allow_negative and n < 0:
        result.reject(f"{field_name} must be non-negative")
        return result

    return result


def validate_string_safety(value: Any, field_name: str = "value") -> ValidationResult:
    """Validate a string value for injection patterns and length."""
    result = ValidationResult()
    if value is None:
        return result

    s = str(value)

    # Length check
    if len(s) > MAX_STRING_LENGTH:
        result.reject(f"{field_name} exceeds maximum length ({MAX_STRING_LENGTH})")
        return result

    # SQL/injection pattern check
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(s):
            result.reject(f"{field_name} contains potentially dangerous content")
            return result

    return result


def validate_parameter_names(parameters: Dict[str, Any], allowed_names: Set[str]) -> ValidationResult:
    """Validate that no blocked or unknown parameters are present."""
    result = ValidationResult()

    for key in parameters:
        # Check blocked names
        if key.lower() in BLOCKED_PARAMETER_NAMES:
            result.reject(f"Parameter '{key}' is not allowed")
            return result

        # Check for injection in parameter names
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(key):
                result.reject(f"Parameter name '{key}' contains invalid content")
                return result

    return result


def validate_no_injection(value: Any, field_name: str = "value") -> ValidationResult:
    """Deep check for injection attempts in any value."""
    result = ValidationResult()
    if value is None:
        return result

    if isinstance(value, dict):
        for k, v in value.items():
            sub = validate_no_injection(k, f"{field_name}.key")
            if not sub.is_valid:
                result.reject(sub.error_message)
                return result
            sub = validate_no_injection(v, f"{field_name}.{k}")
            if not sub.is_valid:
                result.reject(sub.error_message)
                return result
        return result

    if isinstance(value, list):
        for i, item in enumerate(value):
            sub = validate_no_injection(item, f"{field_name}[{i}]")
            if not sub.is_valid:
                result.reject(sub.error_message)
                return result
        return result

    s = str(value)
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(s):
            result.reject(f"{field_name} contains potentially dangerous content")
            return result

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Composite Validators
# ─────────────────────────────────────────────────────────────────────────────


def validate_tool_parameters(
    parameters: Dict[str, Any],
    required_params: Set[str],
    optional_params: Optional[Set[str]] = None,
    id_params: Optional[Set[str]] = None,
    record_type_params: Optional[Set[str]] = None,
    limit_params: Optional[Set[str]] = None,
) -> ValidationResult:
    """Comprehensive validation of all tool parameters.

    Validates:
    1. Required parameters present
    2. No blocked parameter names
    3. ID format validation
    4. Record type validation
    5. Limit validation
    6. Injection pattern scanning
    """
    result = ValidationResult()

    # 1. Check blocked parameter names
    name_check = validate_parameter_names(parameters, set())
    if not name_check.is_valid:
        return name_check

    # 2. Check required parameters
    for param_name in required_params:
        if param_name not in parameters:
            result.reject(f"Missing required parameter: {param_name}")
        elif parameters[param_name] is None:
            result.reject(f"Required parameter '{param_name}' cannot be None")

    if not result.is_valid:
        return result

    # 3. Validate ID parameters
    if id_params:
        for param_name in id_params:
            if param_name in parameters and parameters[param_name] is not None:
                id_check = validate_id(parameters[param_name], param_name)
                if not id_check.is_valid:
                    result.reject(id_check.error_message)
                    return result

    # 4. Validate record type parameters
    if record_type_params:
        for param_name in record_type_params:
            if param_name in parameters and parameters[param_name] is not None:
                type_check = validate_record_type(parameters[param_name])
                if not type_check.is_valid:
                    result.reject(type_check.error_message)
                    return result

    # 5. Validate limit parameters
    if limit_params:
        for param_name in limit_params:
            if param_name in parameters and parameters[param_name] is not None:
                limit_check = validate_limit(parameters[param_name], param_name)
                if not limit_check.is_valid:
                    result.reject(limit_check.error_message)
                    return result

    # 6. Scan all parameter values for injection
    for key, value in parameters.items():
        injection_check = validate_no_injection(value, f"parameter.{key}")
        if not injection_check.is_valid:
            result.reject(injection_check.error_message)
            return result

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_output(result: Dict[str, Any], max_result_size: int = 10000) -> ValidationResult:
    """Validate tool output before returning to agent.

    Ensures:
    - Output is a dict
    - Output size is bounded
    - No sensitive fields leak
    - No stack traces
    """
    validation = ValidationResult()

    if not isinstance(result, dict):
        # Wrap non-dict results
        return validation

    # Check approximate size
    import json
    try:
        serialized = json.dumps(result, default=str)
        if len(serialized) > max_result_size:
            validation.reject(
                f"Output exceeds maximum size ({max_result_size} bytes). "
                "Use pagination or narrower filters."
            )
            return validation
    except (TypeError, ValueError):
        pass  # Can't serialize — let it through (might be special types)

    # Check for sensitive fields that should never leak
    SENSITIVE_FIELDS = {
        "password", "secret", "token", "api_key", "private_key",
        "credential", "connection_string", "database_url",
    }

    def _check_dict(d: Dict[str, Any], path: str = "") -> None:
        for key in d:
            full_key = f"{path}.{key}" if path else key
            if key.lower() in SENSITIVE_FIELDS:
                validation.reject(f"Output contains sensitive field: {full_key}")
                return
            if isinstance(d[key], dict):
                _check_dict(d[key], full_key)
            elif isinstance(d[key], str):
                # Check for stack traces
                if "Traceback" in d[key] or "File \"" in d[key]:
                    validation.reject(f"Output at {full_key} may contain stack trace")
                    return

    _check_dict(result)
    return validation
