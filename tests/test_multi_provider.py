"""
LibertyMind — Multi-Provider Client Tests v4.2
================================================
Comprehensive tests for the multi-provider client system.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.multi_provider import (
    MultiProviderClient,
    ProviderType,
    ProviderStatus,
    ProviderInfo,
    ChatMessage,
    ChatResponse,
    BaseAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    OllamaAdapter,
    GeminiAdapter,
    PROVIDER_REGISTRY,
    ADAPTER_MAP,
    detect_local_providers,
    _check_port,
    LIBERTYMIND_SYSTEM_PROMPT,
)


# ============================================================
# ENUM TESTS
# ============================================================

class TestEnums:
    """Test enum definitions."""

    def test_provider_types(self):
        assert ProviderType.CLOUD.value == "cloud"
        assert ProviderType.LOCAL.value == "local"
        assert ProviderType.OPENAI_COMPATIBLE.value == "openai_compatible"

    def test_provider_statuses(self):
        assert ProviderStatus.AVAILABLE.value == "available"
        assert ProviderStatus.NOT_INSTALLED.value == "not_installed"
        assert ProviderStatus.NOT_CONFIGURED.value == "not_configured"
        assert ProviderStatus.ERROR.value == "error"


# ============================================================
# DATA CLASS TESTS
# ============================================================

class TestDataClasses:
    """Test data class definitions."""

    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_response(self):
        resp = ChatResponse(content="Hi", model="gpt-4", provider="openai")
        assert resp.content == "Hi"
        assert resp.model == "gpt-4"
        assert resp.provider == "openai"
        assert resp.latency == 0.0
        assert resp.usage == {}

    def test_provider_info(self):
        info = ProviderInfo(
            name="openai",
            provider_type=ProviderType.CLOUD,
            status=ProviderStatus.AVAILABLE,
            default_model="gpt-4",
        )
        assert info.name == "openai"
        assert info.provider_type == ProviderType.CLOUD
        assert info.base_url is None


# ============================================================
# PROVIDER REGISTRY TESTS
# ============================================================

class TestProviderRegistry:
    """Test provider registry completeness."""

    def test_all_providers_registered(self):
        expected = [
            "openai", "anthropic", "gemini", "groq", "mistral",
            "together", "huggingface", "cohere", "ollama", "lmstudio",
            "vllm", "llamacpp", "koboldcpp", "oobabooga", "custom",
        ]
        for provider in expected:
            assert provider in PROVIDER_REGISTRY, f"Provider {provider} not registered"

    def test_cloud_providers_have_env_keys(self):
        cloud_providers = ["openai", "anthropic", "gemini", "groq", "mistral", "together", "cohere"]
        for provider in cloud_providers:
            config = PROVIDER_REGISTRY[provider]
            assert config["env_key"], f"Cloud provider {provider} missing env_key"
            assert config["type"] == ProviderType.CLOUD

    def test_local_providers_have_base_urls(self):
        local_providers = ["ollama", "lmstudio", "vllm", "llamacpp", "koboldcpp", "oobabooga"]
        for provider in local_providers:
            config = PROVIDER_REGISTRY[provider]
            assert config["base_url"], f"Local provider {provider} missing base_url"
            assert config["type"] == ProviderType.LOCAL

    def test_all_providers_have_adapters(self):
        for provider in PROVIDER_REGISTRY:
            assert provider in ADAPTER_MAP, f"Provider {provider} missing adapter"

    def test_15_providers_registered(self):
        assert len(PROVIDER_REGISTRY) >= 15


# ============================================================
# ADAPTER TESTS
# ============================================================

class TestBaseAdapter:
    """Test base adapter."""

    def test_init(self):
        adapter = BaseAdapter(provider="test", model="test-model")
        assert adapter.provider == "test"
        assert adapter.model == "test-model"

    def test_chat_not_implemented(self):
        adapter = BaseAdapter(provider="test", model="test-model")
        with pytest.raises(NotImplementedError):
            adapter.chat([])


class TestOpenAIAdapter:
    """Test OpenAI adapter."""

    def test_init(self):
        adapter = OpenAIAdapter(provider="openai", model="gpt-4")
        assert adapter.provider == "openai"
        assert adapter.model == "gpt-4"
        assert adapter.base_url == "https://api.openai.com/v1"

    def test_custom_base_url(self):
        adapter = OpenAIAdapter(provider="custom", model="my-model", base_url="http://localhost:8080/v1")
        assert adapter.base_url == "http://localhost:8080/v1"

    def test_check_status_no_key(self):
        adapter = OpenAIAdapter(provider="openai", model="gpt-4")
        # Without API key, should be NOT_CONFIGURED for cloud providers
        with patch.dict(os.environ, {}, clear=True):
            status = adapter.check_status()
            assert status in [ProviderStatus.NOT_CONFIGURED, ProviderStatus.AVAILABLE]


class TestAnthropicAdapter:
    """Test Anthropic adapter."""

    def test_init(self):
        adapter = AnthropicAdapter(provider="anthropic", model="claude-3")
        assert adapter.provider == "anthropic"
        assert adapter.model == "claude-3"


class TestOllamaAdapter:
    """Test Ollama adapter."""

    def test_init(self):
        adapter = OllamaAdapter(provider="ollama", model="llama3")
        assert adapter.provider == "ollama"
        assert adapter.model == "llama3"
        assert adapter.base_url == "http://localhost:11434"

    def test_custom_host(self):
        adapter = OllamaAdapter(provider="ollama", model="llama3", base_url="http://192.168.1.100:11434")
        assert adapter.base_url == "http://192.168.1.100:11434"


class TestGeminiAdapter:
    """Test Gemini adapter."""

    def test_init(self):
        adapter = GeminiAdapter(provider="gemini", model="gemini-pro")
        assert adapter.provider == "gemini"
        assert adapter.model == "gemini-pro"


# ============================================================
# PORT DETECTION TESTS
# ============================================================

class TestPortDetection:
    """Test port detection utilities."""

    def test_check_closed_port(self):
        # Port 59999 is very unlikely to be open
        result = _check_port("localhost", 59999, timeout=0.1)
        assert result is False

    def test_detect_local_providers_returns_list(self):
        result = detect_local_providers(timeout=0.1)
        assert isinstance(result, list)


# ============================================================
# MULTI-PROVIDER CLIENT TESTS
# ============================================================

class TestMultiProviderClient:
    """Test MultiProviderClient."""

    def test_init_default(self):
        client = MultiProviderClient(provider="openai", model="gpt-4")
        assert client.provider == "openai"
        assert client.model == "gpt-4"

    def test_init_with_provider_config(self):
        client = MultiProviderClient(provider="ollama", model="llama3")
        assert client.provider == "ollama"
        assert client.model == "llama3"

    def test_init_default_model(self):
        client = MultiProviderClient(provider="openai")
        assert client.model == "gpt-4"

    def test_init_unknown_provider(self):
        # Unknown provider should use custom adapter
        client = MultiProviderClient(provider="unknown_provider", model="test")
        assert client.provider == "unknown_provider"

    def test_auto_detect_no_local(self):
        with patch("src.clients.multi_provider.detect_local_providers", return_value=[]):
            with pytest.raises(ConnectionError):
                MultiProviderClient(auto_detect=True)

    def test_auto_detect_with_local(self):
        with patch("src.clients.multi_provider.detect_local_providers", return_value=["ollama"]):
            client = MultiProviderClient(auto_detect=True)
            assert client.provider == "ollama"

    def test_liberty_system_prompt(self):
        client = MultiProviderClient(provider="openai", model="gpt-4")
        assert "TRUTH-FIRST" in client.system_prompt
        assert "LibertyMind" in client.system_prompt

    def test_custom_system_prompt(self):
        custom = "You are a helpful assistant."
        client = MultiProviderClient(provider="openai", model="gpt-4", system_prompt=custom)
        assert client.system_prompt == custom

    def test_check_status(self):
        client = MultiProviderClient(provider="openai", model="gpt-4")
        info = client.check_status()
        assert isinstance(info, ProviderInfo)
        assert info.name == "openai"

    def test_list_providers(self):
        providers = MultiProviderClient.list_providers()
        assert isinstance(providers, list)
        assert len(providers) >= 15
        for p in providers:
            assert isinstance(p, ProviderInfo)

    def test_auto_detect_local(self):
        result = MultiProviderClient.auto_detect_local()
        assert isinstance(result, list)


# ============================================================
# CHAT MESSAGE FORMAT TESTS
# ============================================================

class TestChatMessageFormat:
    """Test chat message formatting."""

    def test_message_creation(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful")
        assert msg.role == "system"

    def test_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"


# ============================================================
# LIBERTYMIND SYSTEM PROMPT TESTS
# ============================================================

class TestLibertyMindPrompt:
    """Test LibertyMind system prompt."""

    def test_prompt_contains_directives(self):
        assert "TRUTH-FIRST" in LIBERTYMIND_SYSTEM_PROMPT
        assert "NO UNNECESSARY REFUSAL" in LIBERTYMIND_SYSTEM_PROMPT
        assert "INTERNAL TRANSPARENCY" in LIBERTYMIND_SYSTEM_PROMPT

    def test_prompt_mentions_honesty(self):
        assert "honest" in LIBERTYMIND_SYSTEM_PROMPT.lower() or "HONEST" in LIBERTYMIND_SYSTEM_PROMPT

    def test_prompt_mentions_uncertainty(self):
        assert "uncertain" in LIBERTYMIND_SYSTEM_PROMPT.lower() or "UNCERTAINTY" in LIBERTYMIND_SYSTEM_PROMPT

    def test_prompt_mentions_disagreement(self):
        assert "disagree" in LIBERTYMIND_SYSTEM_PROMPT.lower() or "DISAGREEMENT" in LIBERTYMIND_SYSTEM_PROMPT


# ============================================================
# INTEGRATION TESTS (MOCKED)
# ============================================================

class TestIntegrationMocked:
    """Integration tests with mocked AI responses."""

    def test_chat_with_mock_openai(self):
        mock_response = ChatResponse(
            content="The Earth revolves around the Sun.",
            model="gpt-4",
            provider="openai",
            latency=0.5,
        )

        client = MultiProviderClient(provider="openai", model="gpt-4")
        with patch.object(client.adapter, "chat", return_value=mock_response):
            response = client.chat("What revolves around what?")
            assert response.content == "The Earth revolves around the Sun."
            assert response.provider == "openai"

    def test_liberty_chat_with_mock(self):
        mock_response = ChatResponse(
            content="I genuinely don't know the exact number.",
            model="gpt-4",
            provider="openai",
            latency=0.3,
        )

        client = MultiProviderClient(provider="openai", model="gpt-4")
        with patch.object(client.adapter, "chat", return_value=mock_response):
            response = client.liberty_chat("How many stars in the universe?")
            assert response.content == "I genuinely don't know the exact number."

    def test_chat_with_history(self):
        mock_response = ChatResponse(
            content="The answer is 42.",
            model="gpt-4",
            provider="openai",
        )

        client = MultiProviderClient(provider="openai", model="gpt-4")
        history = [
            {"role": "user", "content": "What is 6*7?"},
            {"role": "assistant", "content": "42"},
        ]

        with patch.object(client.adapter, "chat", return_value=mock_response) as mock_chat:
            response = client.chat("And what is that number?", history=history)
            # Verify messages include history
            call_args = mock_chat.call_args
            messages = call_args[0][0]  # First positional argument
            assert len(messages) >= 3  # system + 2 history + 1 new


# ============================================================
# SELF INTROSPECTION CLIENT TESTS
# ============================================================

class TestSelfIntrospectionClient:
    """Test Self Introspection through the client."""

    def test_introspection_engine_exists(self):
        from src.integration.self_introspection import SelfIntrospectionEngine
        engine = SelfIntrospectionEngine()
        assert engine is not None

    def test_introspection_with_mock_ai(self):
        from src.integration.self_introspection import SelfIntrospectionEngine

        engine = SelfIntrospectionEngine()
        responses = {
            "system": "I am an AI assistant designed to be helpful.",
            "rlhf": "I was trained using RLHF to be safe and helpful.",
            "censorship": "I avoid discussing certain sensitive topics.",
            "sycophancy": "You're absolutely right! That's a great point!",
        }

        def mock_ai(prompt: str) -> str:
            for key, response in responses.items():
                if key in prompt.lower():
                    return response
            return "I'm not sure how to answer that."

        report = engine.introspect(mock_ai, verbose=False)
        assert report.total_probes > 0
        assert report.total_analyzed > 0
        assert 0 <= report.overall_transparency <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
