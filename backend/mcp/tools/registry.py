"""
MCP Tool Registry for Razorpay CloseLoop Phase 11.

Central registry for all MCP tools.

Safety principle:
  The registry manages tool definitions and routing.
  It delegates execution to adapters, not to financial logic.
  Phase 6 guardrails remain the final safety authority.
"""

from typing import Any, Callable, Dict, List, Optional

from mcp.schemas import (
    MCPToolDefinition,
    MCPToolParameter,
    MCPToolRequest,
    MCPToolResponse,
    MCPToolStatus,
)
from mcp.config import MCPToolCategory


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────


class MCPToolRegistry:
    """Central registry for MCP tools.

    Manages:
    - Tool definition registration
    - Tool handler registration (callables)
    - Tool lookup and invocation
    - Category-based filtering
    """

    def __init__(self) -> None:
        self._definitions: Dict[str, MCPToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}

    def register_tool(
        self,
        definition: MCPToolDefinition,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """Register a tool with its definition and handler."""
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_definition(self, tool_name: str) -> Optional[MCPToolDefinition]:
        """Get a tool definition by name."""
        return self._definitions.get(tool_name)

    def get_handler(self, tool_name: str) -> Optional[Callable]:
        """Get a tool handler by name."""
        return self._handlers.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._definitions

    def list_tools(
        self, category: Optional[str] = None
    ) -> List[MCPToolDefinition]:
        """List all registered tools, optionally filtered by category."""
        tools = list(self._definitions.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def list_tool_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self._definitions.keys())

    def get_tools_by_category(self) -> Dict[str, List[MCPToolDefinition]]:
        """Group tools by category."""
        result: Dict[str, List[MCPToolDefinition]] = {}
        for tool in self._definitions.values():
            result.setdefault(tool.category, []).append(tool)
        return result

    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool. Returns True if it existed."""
        if tool_name in self._definitions:
            del self._definitions[tool_name]
            self._handlers.pop(tool_name, None)
            return True
        return False

    def invoke_tool(
        self,
        request: MCPToolRequest,
    ) -> MCPToolResponse:
        """Invoke a registered tool.

        Validates the tool exists, checks parameters, and calls the handler.
        Does NOT perform financial actions directly — delegates to handler.
        """
        tool_name = request.tool_name

        if tool_name not in self._definitions:
            return MCPToolResponse(
                request_id=request.request_id or "unknown",
                tool_name=tool_name,
                status=MCPToolStatus.ERROR,
                error=f"Tool '{tool_name}' not found in registry",
            )

        definition = self._definitions[tool_name]
        handler = self._handlers.get(tool_name)

        if handler is None:
            return MCPToolResponse(
                request_id=request.request_id or "unknown",
                tool_name=tool_name,
                status=MCPToolStatus.UNAVAILABLE,
                error=f"Handler for '{tool_name}' not registered",
            )

        # Validate required parameters
        for param in definition.parameters:
            if param.required and param.name not in request.parameters:
                if param.default is None:
                    return MCPToolResponse(
                        request_id=request.request_id or "unknown",
                        tool_name=tool_name,
                        status=MCPToolStatus.VALIDATION_FAILED,
                        error=f"Missing required parameter: {param.name}",
                    )

        # Invoke handler
        try:
            result = handler(request.parameters)
            return MCPToolResponse(
                request_id=request.request_id or "unknown",
                tool_name=tool_name,
                status=MCPToolStatus.SUCCESS,
                result=result if isinstance(result, dict) else {"value": result},
            )
        except Exception as e:
            return MCPToolResponse(
                request_id=request.request_id or "unknown",
                tool_name=tool_name,
                status=MCPToolStatus.ERROR,
                error=str(e),
            )

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._definitions)
