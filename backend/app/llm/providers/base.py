"""
Abstract LLM Provider Interface for Razorpay CloseLoop Phase 12A.

Defines the contract that all LLM providers must implement.
The rest of the application depends only on this interface,
never on a specific provider.

IMPORTANT:
This interface is for text generation only.
It does NOT authorize financial actions.
Phase 6 remains the final safety authority.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Provider Type
# ─────────────────────────────────────────────────────────────────────────────


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    NONE = "none"


# ─────────────────────────────────────────────────────────────────────────────
# Message Schema
# ─────────────────────────────────────────────────────────────────────────────


class LLMMessage(BaseModel):
    """A message in the LLM conversation format."""

    role: str = Field(
        ..., description="Message role: system, user, or assistant"
    )
    content: str = Field(..., description="Message content")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response
# ─────────────────────────────────────────────────────────────────────────────


class LLMRequest(BaseModel):
    """Structured LLM request."""

    messages: List[LLMMessage] = Field(
        ..., description="Conversation messages", min_length=1
    )
    model: Optional[str] = Field(
        default=None, description="Model override (uses config default if None)"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature override",
        ge=0.0,
        le=2.0,
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Max tokens override", gt=0
    )
    timeout: Optional[float] = Field(
        default=None, description="Timeout override in seconds", gt=0
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request metadata (workflow_id, exception_id, etc.)",
    )


class LLMResponse(BaseModel):
    """Structured LLM response."""

    content: str = Field(default="", description="Generated text content")
    model: str = Field(default="", description="Model that generated the response")
    provider: str = Field(default="", description="Provider name")
    finish_reason: str = Field(default="", description="Why generation stopped")
    usage: Dict[str, int] = Field(
        default_factory=dict, description="Token usage: prompt_tokens, completion_tokens, total_tokens"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )


class LLMHealthStatus(BaseModel):
    """Health check result for an LLM provider."""

    provider: str = Field(..., description="Provider name")
    healthy: bool = Field(..., description="Whether the provider is healthy")
    model: str = Field(default="", description="Configured model")
    latency_ms: Optional[float] = Field(
        default=None, description="Health check latency in milliseconds"
    )
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional health details"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Provider Interface
# ─────────────────────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All providers must implement:
    - generate(): produce a text completion
    - health_check(): verify the provider is reachable
    - provider_type: identify the provider type

    The provider must NOT:
    - authorize financial actions
    - bypass Phase 6 guardrails
    - access the database directly
    - execute financial operations
    """

    @property
    @abstractmethod
    def provider_type(self) -> LLMProviderType:
        """Return the provider type."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable provider name."""
        ...

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            request: Structured request with messages and parameters.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMProviderError: On provider-specific errors.
            LLMTimeoutError: On timeout.
            LLMConnectionError: On connection failure.
        """
        ...

    @abstractmethod
    async def health_check(self) -> LLMHealthStatus:
        """Check if the provider is reachable and healthy.

        Returns:
            LLMHealthStatus with health information.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Error Types
# ─────────────────────────────────────────────────────────────────────────────


class LLMProviderError(Exception):
    """Base error for LLM provider failures."""

    def __init__(self, message: str, provider: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class LLMTimeoutError(LLMProviderError):
    """Error when LLM request times out."""
    pass


class LLMConnectionError(LLMProviderError):
    """Error when LLM connection fails."""
    pass


class LLMResponseError(LLMProviderError):
    """Error when LLM returns an invalid or unexpected response."""
    pass


class LLMConfigError(LLMProviderError):
    """Error when LLM configuration is invalid."""
    pass
