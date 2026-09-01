"""
LLM Provider Service for Razorpay CloseLoop Phase 12A.

Factory for creating and managing LLM providers.
Provider selection is driven by the LLM_PROVIDER environment variable.

The rest of the application interacts with providers through this service.
Provider-specific logic is isolated behind the abstract interface.

IMPORTANT:
This service creates providers. It does NOT authorize financial actions.
"""

from typing import Optional

from app.llm.config import LLMConfig
from app.llm.providers.base import (
    LLMConfigError,
    LLMProvider,
    LLMProviderType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Provider Factory
# ─────────────────────────────────────────────────────────────────────────────


def create_provider(
    config: Optional[LLMConfig] = None,
) -> Optional[LLMProvider]:
    """Create an LLM provider based on configuration.

    Args:
        config: LLM configuration. Loads from env if not provided.

    Returns:
        Configured LLM provider, or None if LLM is disabled.

    Raises:
        LLMConfigError: If provider type is unknown or config is invalid.
    """
    if config is None:
        config = LLMConfig.from_env()

    if not config.enabled:
        return None

    provider_type = config.provider.lower().strip()

    if provider_type == "openai":
        from app.llm.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(config.openai)
    elif provider_type == "ollama":
        from app.llm.providers.ollama_provider import OllamaProvider
        return OllamaProvider(config.ollama)
    elif provider_type == "none":
        return None
    else:
        raise LLMConfigError(
            f"Unknown LLM provider: '{provider_type}'. "
            f"Supported providers: openai, ollama, none",
            provider=provider_type,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Provider Manager
# ─────────────────────────────────────────────────────────────────────────────


class LLMProviderManager:
    """Manages the application-wide LLM provider instance.

    Ensures a single provider instance is created and reused.
    Thread-safe for concurrent access.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the provider manager.

        Args:
            config: LLM configuration. Loads from env if not provided.
        """
        self._config = config or LLMConfig.from_env()
        self._provider: Optional[LLMProvider] = None

    @property
    def provider(self) -> Optional[LLMProvider]:
        """Get or lazily create the provider."""
        if self._provider is None and self._config.enabled:
            self._provider = create_provider(self._config)
        return self._provider

    @property
    def is_enabled(self) -> bool:
        """Whether LLM is enabled."""
        return self._config.enabled

    @property
    def provider_type(self) -> LLMProviderType:
        """Current provider type."""
        if self._provider is None:
            return LLMProviderType.NONE
        return self._provider.provider_type

    async def health_check(self):
        """Check provider health."""
        if self._provider is None:
            from app.llm.providers.base import LLMHealthStatus
            return LLMHealthStatus(
                provider="none",
                healthy=True,
                model="",
                details={"message": "LLM disabled — no provider to check"},
            )
        return await self._provider.health_check()

    async def close(self) -> None:
        """Close the provider and clean up resources."""
        if self._provider is not None:
            await self._provider.close()
            self._provider = None

    def get_config_summary(self):
        """Get a summary of the current configuration."""
        return {
            "enabled": self._config.enabled,
            "provider": self._config.provider,
            "provider_type": self.provider_type.value,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

# Default manager instance — lazy, created on first access
_manager: Optional[LLMProviderManager] = None


def get_provider(config: Optional[LLMConfig] = None) -> Optional[LLMProvider]:
    """Get the application-wide LLM provider.

    Creates a provider from the default LLMConfig if none exists.

    Args:
        config: Optional config override.

    Returns:
        Configured provider, or None if LLM is disabled.
    """
    global _manager
    if _manager is None:
        _manager = LLMProviderManager(config)
    return _manager.provider
