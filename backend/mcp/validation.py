"""
MCP Validation for Razorpay CloseLoop Phase 11.

Validates MCP tool requests before invocation.

Safety principle:
  Validation ensures correct tool usage.
  It does not authorize financial actions.
"""

from typing import Any, Dict, List, Optional, Tuple

from mcp.schemas import MCPToolDefinition, MCPToolRequest


def validate_request(
    request: MCPToolRequest,
    definition: Optional[MCPToolDefinition],
) -> Tuple[bool, Optional[str]]:
    """Validate an MCP tool request against its definition.

    Returns:
        (is_valid, error_message)
    """
    if definition is None:
        return False, f"Tool '{request.tool_name}' not found"

    # Check required parameters
    for param in definition.parameters:
        if param.required:
            if param.name not in request.parameters:
                return False, f"Missing required parameter: {param.name}"

            value = request.parameters[param.name]
            if value is None:
                return False, f"Required parameter '{param.name}' is None"

            # Check enum constraint
            if param.enum and str(value) not in param.enum:
                return False, (
                    f"Parameter '{param.name}' value '{value}' "
                    f"not in allowed values: {param.enum}"
                )

    # Check for unexpected parameters
    defined_names = {p.name for p in definition.parameters}
    unexpected = set(request.parameters.keys()) - defined_names
    # Allow extra parameters (they'll be passed through)

    return True, None


def validate_parameters(
    parameters: Dict[str, Any],
    definition: MCPToolDefinition,
) -> Tuple[bool, Optional[str]]:
    """Validate just the parameters dict against a tool definition."""
    for param in definition.parameters:
        if param.required and param.name not in parameters:
            return False, f"Missing required parameter: {param.name}"

        if param.name in parameters:
            value = parameters[param.name]
            if param.enum and str(value) not in param.enum:
                return False, (
                    f"Parameter '{param.name}' value '{value}' "
                    f"not in allowed values: {param.enum}"
                )

    return True, None
