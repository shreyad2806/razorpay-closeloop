"""
LLM Configuration for Razorpay CloseLoop Phase 12A.

All configuration is environment-driven.
No secrets are hardcoded.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field


SUPPORTED_PROVIDERS = frozenset({"openai", "ollama", "none"})
MAX_TIMEOUT = 300.0  # 5 minutes max
MIN_TIMEOUT = 0.1   # 100ms min
MAX_MAX_TOKENS = 128000
MAX_RETRIES = 10
MAX_TEMPERATURE = 2.0


def _validate_url(url: str, name: str) -> None:
    """Validate a URL is not empty and has a scheme."""
    if not url or not url.strip():
        raise ValueError(f"{name} must not be empty")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"{name} must start with http:// or https://, got: {url[:30]}")


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI-compatible providers."""

    api_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL",
    )
    api_key: str = Field(
        default="",
        description="API key (from OPENAI_API_KEY env var)",
    )
    model: str = Field(
        default="gpt-3.5-turbo",
        description="Model name",
    )
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds",
        gt=0,
    )
    temperature: float = Field(
        default=0.0,
        description="Generation temperature",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens to generate",
        gt=0,
    )
    max_retries: int = Field(
        default=2,
        description="Maximum retry attempts",
        ge=0,
    )

    def validate_config(self) -> None:
        """Validate the OpenAI configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        _validate_url(self.api_base_url, "api_base_url")
        if not self.model or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout < MIN_TIMEOUT or self.timeout > MAX_TIMEOUT:
            raise ValueError(
                f"timeout must be between {MIN_TIMEOUT} and {MAX_TIMEOUT}, "
                f"got {self.timeout}"
            )
        if self.max_tokens > MAX_MAX_TOKENS:
            raise ValueError(
                f"max_tokens must be <= {MAX_MAX_TOKENS}, got {self.max_tokens}"
            )
        if self.max_retries > MAX_RETRIES:
            raise ValueError(
                f"max_retries must be <= {MAX_RETRIES}, got {self.max_retries}"
            )

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Load configuration from environment variables."""
        return cls(
            api_base_url=os.environ.get(
                "LLM_OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            api_key=os.environ.get("LLM_OPENAI_API_KEY", ""),
            model=os.environ.get("LLM_OPENAI_MODEL", "gpt-3.5-turbo"),
            timeout=float(os.environ.get("LLM_OPENAI_TIMEOUT", "30.0")),
            temperature=float(os.environ.get("LLM_OPENAI_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("LLM_OPENAI_MAX_TOKENS", "1024")),
            max_retries=int(os.environ.get("LLM_OPENAI_MAX_RETRIES", "2")),
        )


class OllamaConfig(BaseModel):
    """Configuration for Ollama/local providers."""

    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    model: str = Field(
        default="llama3.2",
        description="Model name",
    )
    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds",
        gt=0,
    )
    temperature: float = Field(
        default=0.0,
        description="Generation temperature",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens to generate",
        gt=0,
    )
    max_retries: int = Field(
        default=1,
        description="Maximum retry attempts",
        ge=0,
    )

    def validate_config(self) -> None:
        """Validate the Ollama configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        _validate_url(self.base_url, "base_url")
        if not self.model or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout < MIN_TIMEOUT or self.timeout > MAX_TIMEOUT:
            raise ValueError(
                f"timeout must be between {MIN_TIMEOUT} and {MAX_TIMEOUT}, "
                f"got {self.timeout}"
            )
        if self.max_tokens > MAX_MAX_TOKENS:
            raise ValueError(
                f"max_tokens must be <= {MAX_MAX_TOKENS}, got {self.max_tokens}"
            )
        if self.max_retries > MAX_RETRIES:
            raise ValueError(
                f"max_retries must be <= {MAX_RETRIES}, got {self.max_retries}"
            )

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Load configuration from environment variables."""
        return cls(
            base_url=os.environ.get("LLM_OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.environ.get("LLM_OLLAMA_MODEL", "llama3.2"),
            timeout=float(os.environ.get("LLM_OLLAMA_TIMEOUT", "60.0")),
            temperature=float(os.environ.get("LLM_OLLAMA_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("LLM_OLLAMA_MAX_TOKENS", "2048")),
            max_retries=int(os.environ.get("LLM_OLLAMA_MAX_RETRIES", "1")),
        )


class LLMConfig(BaseModel):
    """Top-level LLM configuration.

    Provider selection is driven by the LLM_PROVIDER environment variable:
    - "openai" → OpenAI-compatible provider
    - "ollama" → Ollama/local provider

    The application should continue to work when no LLM provider is configured.
    """

    provider: str = Field(
        default="openai",
        description="Provider name: openai or ollama",
    )
    enabled: bool = Field(
        default=False,
        description="Whether LLM is enabled (disabled by default — opt-in)",
    )
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig,
        description="OpenAI provider configuration",
    )
    ollama: OllamaConfig = Field(
        default_factory=OllamaConfig,
        description="Ollama provider configuration",
    )

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        enabled_raw = os.environ.get("LLM_ENABLED", "false").lower()
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "openai"),
            enabled=enabled_raw in ("true", "1", "yes"),
            openai=OpenAIConfig.from_env(),
            ollama=OllamaConfig.from_env(),
        )

    def validate(self) -> None:
        """Validate the entire LLM configuration.

        Validates:
        - Provider name is supported
        - Provider-specific config is valid

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider: '{self.provider}'. "
                f"Supported: {sorted(SUPPORTED_PROVIDERS)}"
            )
        if self.provider == "openai":
            self.openai.validate_config()
        elif self.provider == "ollama":
            self.ollama.validate_config()

    def get_provider_config(self):
        """Get the config for the selected provider."""
        if self.provider == "openai":
            return self.openai
        elif self.provider == "ollama":
            return self.ollama
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
