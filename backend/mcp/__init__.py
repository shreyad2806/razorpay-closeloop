"""
MCP (Model Context Protocol) integration for Razorpay CloseLoop Phase 11.

Provides controlled tool access for the LangGraph agent to interact with
financial capabilities through a structured MCP interface.

Architecture:
  LangGraph Agent
      ↓
  MCP Client (tool invocation)
      ↓
  MCP Server (tool routing + validation)
      ↓
  Internal Finance Services (Phase 2–10)
      ↓
  Database

Safety principle:
  MCP is an integration/tool-access layer ONLY.
  MCP MUST NOT replace:
  - Deterministic reconciliation
  - Evidence services
  - Phase 6 guardrails
  - Phase 8 verification
  - Model registry
  - Audit logging
"""
