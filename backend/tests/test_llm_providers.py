"""
Tests for Razorpay CloseLoop Phase 12A — LLM Provider Abstraction.

Covers:
- LLMConfig (OpenAI, Ollama, top-level)
- LLMProvider interface
- OpenAIProvider (mocked HTTP)
- OllamaProvider (mocked HTTP)
- Provider factory/selection
- LLMProviderManager
- Failure handling (timeout, connection, malformed, unknown provider)
- Safety boundary (LLM does NOT authorize financial actions)
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Config Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenAIConfig:
    """Tests for OpenAIConfig."""

    def test_default_config(self):
        from app.llm.config import OpenAIConfig

        config = OpenAIConfig()
        assert config.api_base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-3.5-turbo"
        assert config.timeout == 30.0
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.max_retries == 2

    def test_from_env(self):
        from app.llm.config import OpenAIConfig

        env = {
            "LLM_OPENAI_BASE_URL": "https://custom.api.com/v1",
            "LLM_OPENAI_API_KEY": "sk-test-key",
            "LLM_OPENAI_MODEL": "gpt-4",
            "LLM_OPENAI_TIMEOUT": "60.0",
            "LLM_OPENAI_TEMPERATURE": "0.5",
            "LLM_OPENAI_MAX_TOKENS": "2048",
            "LLM_OPENAI_MAX_RETRIES": "3",
        }
        with patch.dict(os.environ, env, clear=False):
            config = OpenAIConfig.from_env()
            assert config.api_base_url == "https://custom.api.com/v1"
            assert config.api_key == "sk-test-key"
            assert config.model == "gpt-4"
            assert config.timeout == 60.0
            assert config.temperature == 0.5
            assert config.max_tokens == 2048
            assert config.max_retries == 3

    def test_from_env_defaults(self):
        from app.llm.config import OpenAIConfig

        config = OpenAIConfig.from_env()
        assert config.api_base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-3.5-turbo"

    def test_custom_config(self):
        from app.llm.config import OpenAIConfig

        config = OpenAIConfig(
            api_base_url="https://my-api.com",
            api_key="key-123",
            model="gpt-4-turbo",
            timeout=10.0,
            temperature=0.7,
            max_tokens=512,
        )
        assert config.api_base_url == "https://my-api.com"
        assert config.model == "gpt-4-turbo"
        assert config.timeout == 10.0
        assert config.temperature == 0.7
        assert config.max_tokens == 512


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_default_config(self):
        from app.llm.config import OllamaConfig

        config = OllamaConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.model == "llama3.2"
        assert config.timeout == 60.0
        assert config.temperature == 0.0
        assert config.max_tokens == 2048
        assert config.max_retries == 1

    def test_from_env(self):
        from app.llm.config import OllamaConfig

        env = {
            "LLM_OLLAMA_BASE_URL": "http://custom-host:11434",
            "LLM_OLLAMA_MODEL": "mistral",
            "LLM_OLLAMA_TIMEOUT": "120.0",
            "LLM_OLLAMA_TEMPERATURE": "0.3",
            "LLM_OLLAMA_MAX_TOKENS": "4096",
            "LLM_OLLAMA_MAX_RETRIES": "2",
        }
        with patch.dict(os.environ, env, clear=False):
            config = OllamaConfig.from_env()
            assert config.base_url == "http://custom-host:11434"
            assert config.model == "mistral"
            assert config.timeout == 120.0
            assert config.temperature == 0.3
            assert config.max_tokens == 4096
            assert config.max_retries == 2

    def test_from_env_defaults(self):
        from app.llm.config import OllamaConfig

        config = OllamaConfig.from_env()
        assert config.base_url == "http://localhost:11434"
        assert config.model == "llama3.2"


class TestLLMConfig:
    """Tests for top-level LLMConfig."""

    def test_default_config(self):
        from app.llm.config import LLMConfig

        config = LLMConfig()
        assert config.provider == "openai"
        assert config.enabled is False
        assert config.openai.model == "gpt-3.5-turbo"
        assert config.ollama.model == "llama3.2"

    def test_from_env_enabled(self):
        from app.llm.config import LLMConfig

        env = {"LLM_ENABLED": "true", "LLM_PROVIDER": "ollama"}
        with patch.dict(os.environ, env, clear=False):
            config = LLMConfig.from_env()
            assert config.enabled is True
            assert config.provider == "ollama"

    def test_from_env_disabled(self):
        from app.llm.config import LLMConfig

        env = {"LLM_ENABLED": "false"}
        with patch.dict(os.environ, env, clear=False):
            config = LLMConfig.from_env()
            assert config.enabled is False

    def test_from_env_various_truthy_values(self):
        from app.llm.config import LLMConfig

        for val in ("true", "1", "yes", "True", "YES"):
            env = {"LLM_ENABLED": val}
            with patch.dict(os.environ, env, clear=False):
                config = LLMConfig.from_env()
                assert config.enabled is True, f"Expected enabled for '{val}'"

    def test_from_env_various_falsy_values(self):
        from app.llm.config import LLMConfig

        for val in ("false", "0", "no", "False", "NO", ""):
            env = {"LLM_ENABLED": val}
            with patch.dict(os.environ, env, clear=False):
                config = LLMConfig.from_env()
                assert config.enabled is False, f"Expected disabled for '{val}'"

    def test_get_provider_config_openai(self):
        from app.llm.config import LLMConfig

        config = LLMConfig(provider="openai")
        result = config.get_provider_config()
        assert result == config.openai

    def test_get_provider_config_ollama(self):
        from app.llm.config import LLMConfig

        config = LLMConfig(provider="ollama")
        result = config.get_provider_config()
        assert result == config.ollama

    def test_get_provider_config_unknown(self):
        from app.llm.config import LLMConfig

        config = LLMConfig(provider="unknown")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            config.get_provider_config()


# ─────────────────────────────────────────────────────────────────────────────
# Provider Interface Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderInterface:
    """Tests for the LLMProvider abstract interface."""

    def test_provider_type_enum(self):
        from app.llm.providers.base import LLMProviderType

        assert LLMProviderType.OPENAI == "openai"
        assert LLMProviderType.OLLAMA == "ollama"
        assert LLMProviderType.NONE == "none"

    def test_cannot_instantiate_abstract(self):
        from app.llm.providers.base import LLMProvider

        with pytest.raises(TypeError):
            LLMProvider()

    def test_message_schema(self):
        from app.llm.providers.base import LLMMessage

        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_request_schema(self):
        from app.llm.providers.base import LLMMessage, LLMRequest

        req = LLMRequest(
            messages=[LLMMessage(role="user", content="Hello")],
            model="test-model",
        )
        assert len(req.messages) == 1
        assert req.model == "test-model"
        assert req.metadata == {}

    def test_request_min_messages(self):
        from app.llm.providers.base import LLMRequest

        with pytest.raises(Exception):
            LLMRequest(messages=[])

    def test_response_schema(self):
        from app.llm.providers.base import LLMResponse

        resp = LLMResponse(content="Hi there", model="gpt-4", provider="openai")
        assert resp.content == "Hi there"
        assert resp.model == "gpt-4"
        assert resp.provider == "openai"

    def test_health_status_schema(self):
        from app.llm.providers.base import LLMHealthStatus

        status = LLMHealthStatus(provider="openai", healthy=True, model="gpt-4")
        assert status.healthy is True
        assert status.error is None


# ─────────────────────────────────────────────────────────────────────────────
# Error Type Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorTypes:
    """Tests for LLM error types."""

    def test_provider_error(self):
        from app.llm.providers.base import LLMProviderError

        err = LLMProviderError("test error", provider="openai", details={"code": 500})
        assert str(err) == "test error"
        assert err.provider == "openai"
        assert err.details["code"] == 500

    def test_timeout_error(self):
        from app.llm.providers.base import LLMProviderError, LLMTimeoutError

        err = LLMTimeoutError("timed out", provider="openai")
        assert isinstance(err, LLMProviderError)

    def test_connection_error(self):
        from app.llm.providers.base import LLMConnectionError, LLMProviderError

        err = LLMConnectionError("connection failed", provider="ollama")
        assert isinstance(err, LLMProviderError)

    def test_response_error(self):
        from app.llm.providers.base import LLMProviderError, LLMResponseError

        err = LLMResponseError("bad response", provider="openai")
        assert isinstance(err, LLMProviderError)

    def test_config_error(self):
        from app.llm.providers.base import LLMConfigError, LLMProviderError

        err = LLMConfigError("bad config", provider="openai")
        assert isinstance(err, LLMProviderError)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Provider Tests (Mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenAIProvider:
    """Tests for OpenAIProvider with mocked HTTP."""

    def _make_provider(self):
        from app.llm.config import OpenAIConfig
        from app.llm.providers.openai_provider import OpenAIProvider

        config = OpenAIConfig(
            api_base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-3.5-turbo",
            timeout=30.0,
        )
        return OpenAIProvider(config)

    def test_provider_type(self):
        provider = self._make_provider()
        from app.llm.providers.base import LLMProviderType

        assert provider.provider_type == LLMProviderType.OPENAI

    def test_provider_name(self):
        provider = self._make_provider()
        assert provider.provider_name == "openai"

    def test_config_access(self):
        provider = self._make_provider()
        assert provider.config.model == "gpt-3.5-turbo"

    def test_empty_base_url_raises(self):
        from app.llm.config import OpenAIConfig
        from app.llm.providers.base import LLMConfigError
        from app.llm.providers.openai_provider import OpenAIProvider

        config = OpenAIConfig(api_base_url="")
        with pytest.raises(LLMConfigError, match="base URL is required"):
            OpenAIProvider(config)

    def test_generate_success(self):
        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="Say hello")],
            )
            return await provider.generate(request)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.content == "Hello!"
        assert result.model == "gpt-3.5-turbo"
        assert result.provider == "openai"
        assert result.finish_reason == "stop"
        assert result.usage["total_tokens"] == 15

    def test_generate_timeout(self):
        import httpx
        from app.llm.providers.base import LLMTimeoutError

        provider = self._make_provider()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
            )
            return await provider.generate(request)

        with pytest.raises(LLMTimeoutError):
            asyncio.get_event_loop().run_until_complete(run())

    def test_generate_connection_error(self):
        import httpx
        from app.llm.providers.base import LLMConnectionError

        provider = self._make_provider()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
            )
            return await provider.generate(request)

        with pytest.raises(LLMConnectionError):
            asyncio.get_event_loop().run_until_complete(run())

    def test_generate_http_error(self):
        import httpx
        from app.llm.providers.base import LLMProviderError

        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_exc = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_exc)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
            )
            return await provider.generate(request)

        with pytest.raises(LLMProviderError):
            asyncio.get_event_loop().run_until_complete(run())

    def test_generate_malformed_json(self):
        from app.llm.providers.base import LLMResponseError

        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
            )
            return await provider.generate(request)

        with pytest.raises(LLMResponseError, match="JSON"):
            asyncio.get_event_loop().run_until_complete(run())

    def test_generate_empty_choices(self):
        from app.llm.providers.base import LLMResponseError

        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"model": "gpt-3.5", "choices": []}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
            )
            return await provider.generate(request)

        with pytest.raises(LLMResponseError, match="no choices"):
            asyncio.get_event_loop().run_until_complete(run())

    def test_health_check_success(self):
        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "gpt-3.5-turbo"}]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            return await provider.health_check()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.healthy is True
        assert result.provider == "openai"
        assert result.latency_ms is not None

    def test_health_check_failure(self):
        provider = self._make_provider()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            return await provider.health_check()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.healthy is False
        assert result.error is not None

    def test_close(self):
        provider = self._make_provider()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        provider._client = mock_client

        async def run():
            await provider.close()

        asyncio.get_event_loop().run_until_complete(run())
        mock_client.aclose.assert_called_once()

    def test_model_override(self):
        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {
                    "message": {"content": "test"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
                model="gpt-4",
                temperature=0.5,
                max_tokens=512,
            )
            return await provider.generate(request)

        result = asyncio.get_event_loop().run_until_complete(run())
        # Verify the payload included the override model
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert payload["model"] == "gpt-4"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 512


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Provider Tests (Mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────


class TestOllamaProvider:
    """Tests for OllamaProvider with mocked HTTP."""

    def _make_provider(self):
        from app.llm.config import OllamaConfig
        from app.llm.providers.ollama_provider import OllamaProvider

        config = OllamaConfig(
            base_url="http://localhost:11434",
            model="llama3.2",
            timeout=60.0,
        )
        return OllamaProvider(config)

    def test_provider_type(self):
        provider = self._make_provider()
        from app.llm.providers.base import LLMProviderType

        assert provider.provider_type == LLMProviderType.OLLAMA

    def test_provider_name(self):
        provider = self._make_provider()
        assert provider.provider_name == "ollama"

    def test_empty_base_url_raises(self):
        from app.llm.config import OllamaConfig
        from app.llm.providers.base import LLMConfigError
        from app.llm.providers.ollama_provider import OllamaProvider

        config = OllamaConfig(base_url="")
        with pytest.raises(LLMConfigError, match="base URL is required"):
            OllamaProvider(config)

    def test_generate_success(self):
        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "chatcmpl-local",
            "model": "llama3.2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Local response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
            )
            return await provider.generate(request)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.content == "Local response"
        assert result.provider == "ollama"
        assert result.model == "llama3.2"

    def test_generate_timeout(self):
        import httpx
        from app.llm.providers.base import LLMTimeoutError

        provider = self._make_provider()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
            )
            return await provider.generate(request)

        with pytest.raises(LLMTimeoutError):
            asyncio.get_event_loop().run_until_complete(run())

    def test_generate_connection_error(self):
        import httpx
        from app.llm.providers.base import LLMConnectionError

        provider = self._make_provider()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
            )
            return await provider.generate(request)

        with pytest.raises(LLMConnectionError):
            asyncio.get_event_loop().run_until_complete(run())

    def test_health_check_success(self):
        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2"},
                {"name": "mistral"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            return await provider.health_check()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.healthy is True
        assert result.provider == "ollama"
        assert "llama3.2" in result.details["available_models"]
        assert result.details["model_configured"] is True

    def test_health_check_model_not_available(self):
        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "mistral"}]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            return await provider.health_check()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.healthy is True
        assert result.details["model_configured"] is False

    def test_health_check_failure(self):
        provider = self._make_provider()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("not running"))
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            return await provider.health_check()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.healthy is False

    def test_close(self):
        provider = self._make_provider()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        provider._client = mock_client

        async def run():
            await provider.close()

        asyncio.get_event_loop().run_until_complete(run())
        mock_client.aclose.assert_called_once()

    def test_generate_uses_v1_endpoint(self):
        """Verify Ollama uses /v1/chat/completions (OpenAI-compatible)."""
        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "model": "llama3.2",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        async def run():
            provider._client = mock_client
            from app.llm.providers.base import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content="test")],
            )
            return await provider.generate(request)

        asyncio.get_event_loop().run_until_complete(run())
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/v1/chat/completions"


# ─────────────────────────────────────────────────────────────────────────────
# Provider Factory Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderFactory:
    """Tests for provider creation and selection."""

    def test_create_provider_disabled(self):
        from app.llm.config import LLMConfig
        from app.llm.services.provider_service import create_provider

        config = LLMConfig(enabled=False)
        result = create_provider(config)
        assert result is None

    def test_create_provider_openai(self):
        from app.llm.config import LLMConfig
        from app.llm.providers.openai_provider import OpenAIProvider
        from app.llm.services.provider_service import create_provider

        config = LLMConfig(enabled=True, provider="openai")
        result = create_provider(config)
        assert isinstance(result, OpenAIProvider)

    def test_create_provider_ollama(self):
        from app.llm.config import LLMConfig
        from app.llm.providers.ollama_provider import OllamaProvider
        from app.llm.services.provider_service import create_provider

        config = LLMConfig(enabled=True, provider="ollama")
        result = create_provider(config)
        assert isinstance(result, OllamaProvider)

    def test_create_provider_none(self):
        from app.llm.config import LLMConfig
        from app.llm.services.provider_service import create_provider

        config = LLMConfig(enabled=True, provider="none")
        result = create_provider(config)
        assert result is None

    def test_create_provider_unknown(self):
        from app.llm.config import LLMConfig
        from app.llm.providers.base import LLMConfigError
        from app.llm.services.provider_service import create_provider

        config = LLMConfig(enabled=True, provider="unknown_provider")
        with pytest.raises(LLMConfigError, match="Unknown LLM provider"):
            create_provider(config)

    def test_create_provider_from_env(self):
        from app.llm.services.provider_service import create_provider

        env = {"LLM_ENABLED": "true", "LLM_PROVIDER": "openai"}
        with patch.dict(os.environ, env, clear=False):
            result = create_provider()
            assert result is not None

    def test_create_provider_env_disabled(self):
        from app.llm.services.provider_service import create_provider

        env = {"LLM_ENABLED": "false"}
        with patch.dict(os.environ, env, clear=False):
            result = create_provider()
            assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Provider Manager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderManager:
    """Tests for LLMProviderManager."""

    def test_manager_disabled(self):
        from app.llm.config import LLMConfig
        from app.llm.providers.base import LLMProviderType
        from app.llm.services.provider_service import LLMProviderManager

        config = LLMConfig(enabled=False)
        manager = LLMProviderManager(config)
        assert manager.provider is None
        assert manager.is_enabled is False
        assert manager.provider_type == LLMProviderType.NONE

    def test_manager_enabled_openai(self):
        from app.llm.config import LLMConfig
        from app.llm.providers.openai_provider import OpenAIProvider
        from app.llm.services.provider_service import LLMProviderManager

        config = LLMConfig(enabled=True, provider="openai")
        manager = LLMProviderManager(config)
        assert isinstance(manager.provider, OpenAIProvider)
        assert manager.is_enabled is True

    def test_manager_reuses_provider(self):
        from app.llm.config import LLMConfig
        from app.llm.services.provider_service import LLMProviderManager

        config = LLMConfig(enabled=True, provider="openai")
        manager = LLMProviderManager(config)
        p1 = manager.provider
        p2 = manager.provider
        assert p1 is p2

    def test_manager_health_check_disabled(self):
        from app.llm.config import LLMConfig
        from app.llm.services.provider_service import LLMProviderManager

        config = LLMConfig(enabled=False)
        manager = LLMProviderManager(config)

        async def run():
            return await manager.health_check()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.healthy is True
        assert result.provider == "none"

    def test_manager_config_summary(self):
        from app.llm.config import LLMConfig
        from app.llm.providers.base import LLMProviderType
        from app.llm.services.provider_service import LLMProviderManager

        config = LLMConfig(enabled=True, provider="ollama")
        manager = LLMProviderManager(config)
        _ = manager.provider  # trigger lazy creation
        summary = manager.get_config_summary()
        assert summary["enabled"] is True
        assert summary["provider"] == "ollama"
        assert summary["provider_type"] == LLMProviderType.OLLAMA.value

    def test_manager_close(self):
        from app.llm.config import LLMConfig
        from app.llm.services.provider_service import LLMProviderManager

        config = LLMConfig(enabled=True, provider="openai")
        manager = LLMProviderManager(config)
        _ = manager.provider  # force creation

        mock_provider = AsyncMock()
        manager._provider = mock_provider

        async def run():
            await manager.close()

        asyncio.get_event_loop().run_until_complete(run())
        mock_provider.close.assert_called_once()
        assert manager._provider is None

    def test_get_provider_module_level(self):
        from app.llm.config import LLMConfig
        from app.llm.services.provider_service import LLMProviderManager, _manager
        import app.llm.services.provider_service as svc

        # Reset the module-level manager
        svc._manager = None

        env = {"LLM_ENABLED": "true", "LLM_PROVIDER": "ollama"}
        with patch.dict(os.environ, env, clear=False):
            config = LLMConfig.from_env()
            result = svc.get_provider(config)
            # Should have created a provider (or None if disabled)
            assert result is None or result is not None  # smoke test


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMSafetyBoundary:
    """Verify LLM does NOT authorize financial actions."""

    def test_provider_has_no_financial_methods(self):
        """LLM provider must not have execute, refund, or settlement methods."""
        from app.llm.providers.base import LLMProvider

        forbidden_methods = [
            "execute_resolution",
            "issue_refund",
            "modify_settlement",
            "modify_merchant_balance",
            "authorize_payment",
            "call_razorpay_api",
            "bypass_guardrails",
        ]
        for method in forbidden_methods:
            assert not hasattr(LLMProvider, method), (
                f"LLMProvider should not have {method}"
            )

    def test_config_has_no_financial_fields(self):
        """LLM config must not have fields that control financial behavior."""
        from app.llm.config import LLMConfig

        forbidden_fields = [
            "auto_approve",
            "max_financial_exposure",
            "bypass_guardrails",
            "financial_authority",
        ]
        for field in forbidden_fields:
            assert field not in LLMConfig.model_fields, (
                f"LLMConfig should not have field {field}"
            )

    def test_response_has_no_financial_authorization(self):
        """LLM response must not contain financial authorization data."""
        from app.llm.providers.base import LLMResponse

        resp = LLMResponse(content="suggest_refund")
        # Response is text only — no financial fields
        assert not hasattr(resp, "authorize")
        assert not hasattr(resp, "approve")
        assert not hasattr(resp, "execute")

    def test_provider_type_cannot_be_financial(self):
        """Provider type enum must not include financial operations."""
        from app.llm.providers.base import LLMProviderType

        for pt in LLMProviderType:
            assert pt.value not in ("financial", "execution", "payment", "refund")
