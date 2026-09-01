"""
MCP Configuration for Razorpay CloseLoop Phase 11.

All configuration is environment-driven.
No hardcoded secrets.

Safety principle:
  MCP configuration controls tool access.
  It never authorizes financial execution directly.
"""

import os
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class MCPServerMode(str, Enum):
    """MCP server operating modes."""
    STANDALONE = "standalone"        # Independent MCP server
    EMBEDDED = "embedded"            # Embedded within LangGraph process
    HTTP = "http"                    # HTTP-based MCP server


class MCPToolCategory(str, Enum):
    """Categories of MCP tools."""
    RECONCILIATION = "reconciliation"
    EVIDENCE = "evidence"
    CLASSIFICATION = "classification"
    RESOLUTION = "resolution"
    GUARDRAILS = "guardrails"
    EXECUTION = "execution"
    FEEDBACK = "feedback"
    AUDIT = "audit"
    LINEAGE = "lineage"
    COMPARISON = "comparison"


class MCPServerConfig(BaseModel):
    """MCP server configuration.

    All values are configurable via environment variables.
    """
    # Server settings
    server_name: str = Field(
        default="razorpay-closeloop-mcp",
        description="MCP server name",
    )
    mode: MCPServerMode = Field(
        default=MCPServerMode.EMBEDDED,
        description="Server operating mode",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Server host (for HTTP mode)",
    )
    port: int = Field(
        default=8080,
        description="Server port (for HTTP mode)",
    )

    # Tool access control
    enabled_categories: List[MCPToolCategory] = Field(
        default_factory=lambda: list(MCPToolCategory),
        description="Enabled tool categories",
    )
    disabled_tools: List[str] = Field(
        default_factory=list,
        description="Specific tool names to disable",
    )

    # Safety settings
    require_guardrail_approval: bool = Field(
        default=True,
        description="Require guardrail approval for financial actions",
    )
    require_verification: bool = Field(
        default=True,
        description="Require verification after execution",
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum concurrent tool invocations",
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        description="Tool invocation timeout",
    )

    # Audit settings
    audit_all_requests: bool = Field(
        default=True,
        description="Audit all MCP requests",
    )
    audit_response_bodies: bool = Field(
        default=False,
        description="Audit full response bodies (may contain sensitive data)",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="MCP server log level",
    )

    @classmethod
    def from_env(cls) -> "MCPServerConfig":
        """Load configuration from environment variables."""
        return cls(
            server_name=os.environ.get("MCP_SERVER_NAME", "razorpay-closeloop-mcp"),
            mode=MCPServerMode(os.environ.get("MCP_MODE", "embedded")),
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8080")),
            require_guardrail_approval=os.environ.get(
                "MCP_REQUIRE_GUARDRAIL", "true"
            ).lower() == "true",
            require_verification=os.environ.get(
                "MCP_REQUIRE_VERIFICATION", "true"
            ).lower() == "true",
            max_concurrent_requests=int(
                os.environ.get("MCP_MAX_CONCURRENT", "10")
            ),
            request_timeout_seconds=float(
                os.environ.get("MCP_TIMEOUT", "30.0")
            ),
            audit_all_requests=os.environ.get(
                "MCP_AUDIT_ALL", "true"
            ).lower() == "true",
            log_level=os.environ.get("MCP_LOG_LEVEL", "INFO"),
        )

    class Config:
        frozen = True
