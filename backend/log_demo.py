"""
Demonstrate LangGraph + MCP structured logging.

Runs one complete workflow and prints the resulting log sequence
reconstructed from the execution log + Python logger output.
"""
import logging
import sys
import time

# Configure structured logging to print to stdout
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    stream=sys.stdout,
)

from app.agent.workflow import run_workflow, get_last_execution_log


def main():
    print("=" * 80)
    print("LANGGRAPH WORKFLOW OBSERVABILITY DEMO")
    print("=" * 80)
    print()

    # Run a complete workflow for a known exception
    exception_id = "EXC-001"
    print(f"Running workflow for exception: {exception_id}")
    print()

    start_time = time.perf_counter()
    try:
        result = run_workflow(exception_id=exception_id, batch_id="BATCH-DEMO-001")
        total_ms = (time.perf_counter() - start_time) * 1000
    except Exception as e:
        total_ms = (time.perf_counter() - start_time) * 1000
        print(f"\n  WORKFLOW RAISED: {type(e).__name__}: {e}")
        return

    print()
    print("=" * 80)
    print("WORKFLOW RESULT")
    print("=" * 80)
    print(f"  Decision:      {result.decision}")
    print(f"  Confidence:    {result.confidence}")
    print(f"  Risk:          {result.risk}")
    print(f"  Status:        {result.metadata.workflow_status.value}")
    print(f"  Nodes executed: {len(result.metadata.nodes_executed)}")
    print(f"  Total time:    {round(total_ms, 1)}ms")

    # Retrieve the execution log
    exec_log = get_last_execution_log()
    if exec_log:
        print()
        print("=" * 80)
        print("EXECUTION TRACE (from WorkflowExecutionLog)")
        print("=" * 80)
        summary = exec_log.summary()
        print(f"  Total events:     {summary['total_events']}")
        print(f"  Nodes in order:   {summary['nodes_in_order']}")
        print(f"  Failed nodes:     {summary['failed_nodes']}")
        print(f"  Total time:       {summary['total_ms']}ms")
        print(f"  Final decision:   {summary['final_decision']}")
        print(f"  Final risk:       {summary['final_risk']}")
        print(f"  Final confidence: {summary['final_confidence']}")
        print(f"  Guardrail:        {summary['guardrail_decision']}")
        print(f"  Verification:     {summary['verification_result']}")
        print()
        print("  Per-node timings:")
        for node, ms in summary["node_timings_ms"].items():
            print(f"    {node:35s}  {ms:>8.1f}ms")
        print()

    # Also demo MCP tool logging
    print("=" * 80)
    print("MCP TOOL LOGGING DEMO")
    print("=" * 80)

    from mcp.server import MCPServer
    from mcp.schemas import MCPToolRequest
    from mcp.tools.registry import MCPToolRegistry

    server = MCPServer()
    print(f"  MCP server initialized: {server.config.server_name}")
    print(f"  Tools registered:       {server.registry.tool_count}")

    # Try calling a registered tool
    if server.registry.tool_count > 0:
        tool_name = list(server.registry._tools.keys())[0]
        print(f"\n  Calling MCP tool: {tool_name}")
        req = MCPToolRequest(
            tool_name=tool_name,
            parameters={},
            workflow_id="WF-LOG-DEMO",
            exception_id="EXC-001",
        )
        resp = server.invoke(req)
        print(f"  Response status: {resp.status.value}")
        print(f"  Duration:        {resp.duration_ms:.1f}ms")
    else:
        print("  No MCP tools registered — skipping tool invocation demo")

    print()
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
