"""MCP logging demo — registers tools and invokes them."""
import logging
import sys

logging.basicConfig(level=logging.DEBUG, format="%(message)s", stream=sys.stdout)

from mcp.server import MCPServer
from mcp.config import MCPServerConfig, MCPServerMode
from mcp.schemas import MCPToolRequest, MCPToolDefinition
from mcp.tools.registry import MCPToolRegistry

# Create a server with embedded mode and all categories enabled
config = MCPServerConfig(mode=MCPServerMode.EMBEDDED)
server = MCPServer(config)

# Register a test tool
def _handle_reconcile(params):
    return {"status": "ok", "matched": True, "records": 42}

reconcile_def = MCPToolDefinition(
    name="reconcile_batch",
    description="Run reconciliation on a batch",
    category="reconciliation",
    is_financial=True,
    requires_guardrail=True,
    input_schema={},
)

server.register_tool(reconcile_def, _handle_reconcile)
print(f"\nTools registered: {server.registry.tool_count}")

# Invoke the tool
print("\n--- Invoking reconcile_batch ---")
req = MCPToolRequest(
    tool_name="reconcile_batch",
    parameters={"batch_id": "BATCH-001"},
    workflow_id="WF-MCP-LOG-DEMO",
    exception_id="EXC-001",
)
resp = server.invoke(req)
print(f"Response: status={resp.status.value} duration={resp.duration_ms:.1f}ms")

# Try a tool that doesn't exist
print("\n--- Invoking nonexistent tool ---")
req2 = MCPToolRequest(
    tool_name="nonexistent_tool",
    parameters={},
    workflow_id="WF-MCP-LOG-DEMO",
    exception_id="EXC-001",
)
resp2 = server.invoke(req2)
print(f"Response: status={resp2.status.value} error={resp2.error}")

# Verify audit log
audit = server.get_audit_log()
print(f"\nAudit log entries: {len(audit)}")
for entry in audit:
    print(f"  [{entry.status.value}] {entry.tool_name} ({entry.duration_ms:.1f}ms)")
