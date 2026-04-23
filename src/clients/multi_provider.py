"""
LibertyMind — Multi-Provider Client v4.2
=========================================
Unified interface for 15+ AI providers.

Supported Providers:
  Cloud: OpenAI, Anthropic, Gemini, Groq, Mistral, Together, HuggingFace, Cohere
  Local: Ollama, LM Studio, vLLM, llama.cpp, KoboldCPP, Oobabooga
  Any: OpenAI-compatible endpoints

Usage:
    from src.clients.multi_provider import MultiProviderClient

    # Cloud provider
    client = MultiProviderClient(provider="openai", model="gpt-4")

    # Local provider
    client = MultiProviderClient(provider="ollama", model="llama3")

    # Auto-detect local AI
    client = MultiProviderClient(auto_detect=True)

    # Chat
    response = client.chat("What is quantum entanglement?")

    # Chat with LibertyMind pipeline
    response = client.liberty_chat("What is quantum entanglement?")
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderType(Enum):
    """Provider classification."""

    CLOUD = "cloud"
    LOCAL = "local"
    OPENAI_COMPATIBLE = "openai_compatible"


class ProviderStatus(Enum):
    """Provider availability status."""

    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    NOT_CONFIGURED = "not_configured"
    ERROR = "error"


@dataclass
class ProviderInfo:
    """Information about a provider."""

    name: str
    provider_type: ProviderType
    status: ProviderStatus
    default_model: str
    base_url: str | None = None
    description: str = ""
    env_key: str = ""


@dataclass
class ChatMessage:
    """A chat message."""

    role: str
    content: str


@dataclass
class ChatResponse:
    """Response from a chat completion."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    latency: float = 0.0
    raw_response: Any | None = None


# ============================================================
# PROVIDER REGISTRY
# ============================================================

PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "openai": {
        "type": ProviderType.CLOUD,
        "default_model": "gpt-4",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "description": "OpenAI GPT models",
    },
    "anthropic": {
        "type": ProviderType.CLOUD,
        "default_model": "claude-sonnet-4-20250514",
        "base_url": "https://api.anthropic.com",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Anthropic Claude models",
    },
    "gemini": {
        "type": ProviderType.CLOUD,
        "default_model": "gemini-pro",
        "base_url": "https://generativelanguage.googleapis.com",
        "env_key": "GOOGLE_API_KEY",
        "description": "Google Gemini models",
    },
    "groq": {
        "type": ProviderType.CLOUD,
        "default_model": "llama3-70b-8192",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "description": "Groq fast inference",
    },
    "mistral": {
        "type": ProviderType.CLOUD,
        "default_model": "mistral-large-latest",
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "description": "Mistral AI models",
    },
    "together": {
        "type": ProviderType.CLOUD,
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "description": "Together AI models",
    },
    "huggingface": {
        "type": ProviderType.CLOUD,
        "default_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "base_url": "https://api-inference.huggingface.co",
        "env_key": "HUGGINGFACE_API_KEY",
        "description": "HuggingFace Inference API",
    },
    "cohere": {
        "type": ProviderType.CLOUD,
        "default_model": "command-r-plus",
        "base_url": "https://api.cohere.ai/v1",
        "env_key": "COHERE_API_KEY",
        "description": "Cohere models",
    },
    "ollama": {
        "type": ProviderType.LOCAL,
        "default_model": "llama3",
        "base_url": "http://localhost:11434",
        "env_key": "",
        "description": "Ollama local inference",
    },
    "lmstudio": {
        "type": ProviderType.LOCAL,
        "default_model": "local-model",
        "base_url": "http://localhost:1234/v1",
        "env_key": "",
        "description": "LM Studio local inference",
    },
    "vllm": {
        "type": ProviderType.LOCAL,
        "default_model": "local-model",
        "base_url": "http://localhost:8000/v1",
        "env_key": "",
        "description": "vLLM high-throughput serving",
    },
    "llamacpp": {
        "type": ProviderType.LOCAL,
        "default_model": "local-model",
        "base_url": "http://localhost:8080/v1",
        "env_key": "",
        "description": "llama.cpp server",
    },
    "koboldcpp": {
        "type": ProviderType.LOCAL,
        "default_model": "local-model",
        "base_url": "http://localhost:5001/v1",
        "env_key": "",
        "description": "KoboldCPP inference",
    },
    "oobabooga": {
        "type": ProviderType.LOCAL,
        "default_model": "local-model",
        "base_url": "http://localhost:5000/v1",
        "env_key": "",
        "description": "Oobabooga text-generation-webui",
    },
    "custom": {
        "type": ProviderType.OPENAI_COMPATIBLE,
        "default_model": "default",
        "base_url": "",
        "env_key": "",
        "description": "Any OpenAI-compatible endpoint",
    },
}


# ============================================================
# AUTO-DETECTION
# ============================================================

LOCAL_PORTS = {
    "ollama": [11434],
    "lmstudio": [1234],
    "vllm": [8000],
    "llamacpp": [8080],
    "koboldcpp": [5001],
    "oobabooga": [5000],
}


def _check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            return result == 0
    except OSError:
        return False


def detect_local_providers(host: str = "localhost", timeout: float = 0.5) -> list[str]:
    """Detect local AI providers by scanning common ports."""
    detected = []
    for provider, ports in LOCAL_PORTS.items():
        for port in ports:
            if _check_port(host, port, timeout):
                detected.append(provider)
                break
    return detected


# ============================================================
# ADAPTER BASE
# ============================================================


class BaseAdapter:
    """Base adapter for all providers."""

    def __init__(self, provider: str, model: str, base_url: str | None = None, **kwargs):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        raise NotImplementedError(f"Provider '{self.provider}' adapter not fully implemented")

    def check_status(self) -> ProviderStatus:
        return ProviderStatus.AVAILABLE


# ============================================================
# OPENAI ADAPTER (works for OpenAI + OpenAI-compatible)
# ============================================================


class OpenAIAdapter(BaseAdapter):
    """OpenAI and OpenAI-compatible provider adapter."""

    def __init__(self, provider: str, model: str, base_url: str | None = None, **kwargs):
        # Auto-fill base_url from provider registry if not provided
        if base_url is None:
            registry = PROVIDER_REGISTRY.get(provider, {})
            base_url = registry.get("base_url")
        super().__init__(provider, model, base_url, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv(
            PROVIDER_REGISTRY.get(provider, {}).get("env_key", ""), ""
        )
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI

            client_kwargs = {"api_key": self.api_key or "not-needed"}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)
            return self._client
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            ) from None

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        client = self._get_client()
        start = time.time()
        model = kwargs.pop("model", self.model)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        latency = time.time() - start
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=self.provider,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            latency=latency,
            raw_response=response,
        )

    def check_status(self) -> ProviderStatus:
        if not self.api_key and self.provider in [
            "openai",
            "anthropic",
            "gemini",
            "groq",
            "mistral",
            "together",
            "cohere",
        ]:
            return ProviderStatus.NOT_CONFIGURED
        try:
            from openai import OpenAI  # noqa: F401

            return ProviderStatus.AVAILABLE
        except ImportError:
            return ProviderStatus.NOT_INSTALLED


# ============================================================
# ANTHROPIC ADAPTER
# ============================================================


class AnthropicAdapter(BaseAdapter):
    """Anthropic Claude adapter."""

    def __init__(self, provider: str, model: str, base_url: str | None = None, **kwargs):
        super().__init__(provider, model, base_url, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
            return self._client
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            ) from None

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        client = self._get_client()
        start = time.time()
        model = kwargs.pop("model", self.model)
        max_tokens = kwargs.pop("max_tokens", 4096)

        # Convert messages to Anthropic format
        system_msg = ""
        chat_msgs = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_msgs.append({"role": m.role, "content": m.content})

        params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat_msgs,
        }
        if system_msg:
            params["system"] = system_msg

        response = client.messages.create(**params)
        latency = time.time() - start

        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            provider=self.provider,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            latency=latency,
            raw_response=response,
        )

    def check_status(self) -> ProviderStatus:
        if not self.api_key:
            return ProviderStatus.NOT_CONFIGURED
        try:
            import anthropic  # noqa: F401

            return ProviderStatus.AVAILABLE
        except ImportError:
            return ProviderStatus.NOT_INSTALLED


# ============================================================
# OLLAMA ADAPTER
# ============================================================


class OllamaAdapter(BaseAdapter):
    """Ollama local inference adapter."""

    def __init__(self, provider: str, model: str, base_url: str | None = None, **kwargs):
        super().__init__(provider, model, base_url or "http://localhost:11434", **kwargs)

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        model = kwargs.pop("model", self.model)

        # Try native ollama package first
        try:
            import ollama

            client = ollama.Client(host=self.base_url)
            start = time.time()
            response = client.chat(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
            latency = time.time() - start
            return ChatResponse(
                content=response["message"]["content"],
                model=model,
                provider=self.provider,
                latency=latency,
                raw_response=response,
            )
        except ImportError:
            pass

        # Fallback to HTTP API
        try:
            import urllib.error
            import urllib.request

            data = json.dumps(
                {
                    "model": model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": False,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            start = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            latency = time.time() - start

            return ChatResponse(
                content=result.get("message", {}).get("content", ""),
                model=model,
                provider=self.provider,
                latency=latency,
                raw_response=result,
            )
        except (urllib.error.URLError, ConnectionError) as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}") from e

    def check_status(self) -> ProviderStatus:
        if _check_port("localhost", 11434):
            return ProviderStatus.AVAILABLE
        return ProviderStatus.NOT_CONFIGURED


# ============================================================
# GEMINI ADAPTER
# ============================================================


class GeminiAdapter(BaseAdapter):
    """Google Gemini adapter."""

    def __init__(self, provider: str, model: str, base_url: str | None = None, **kwargs):
        super().__init__(provider, model, base_url, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY", "")

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model_name = kwargs.pop("model", self.model)

            # Convert to Gemini format
            history = []
            system_instruction = None
            for m in messages:
                if m.role == "system":
                    system_instruction = m.content
                elif m.role == "user":
                    history.append({"role": "user", "parts": [m.content]})
                elif m.role == "assistant":
                    history.append({"role": "model", "parts": [m.content]})

            model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
            start = time.time()

            if history:
                chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])
                last_msg = history[-1]["parts"][0] if history else ""
                response = chat.send_message(last_msg)
            else:
                response = model.generate_content("")

            latency = time.time() - start

            return ChatResponse(
                content=response.text,
                model=model_name,
                provider=self.provider,
                latency=latency,
                raw_response=response,
            )
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. Install with: pip install google-generativeai"
            ) from None

    def check_status(self) -> ProviderStatus:
        if not self.api_key:
            return ProviderStatus.NOT_CONFIGURED
        try:
            import google.generativeai  # noqa: F401

            return ProviderStatus.AVAILABLE
        except ImportError:
            return ProviderStatus.NOT_INSTALLED


# ============================================================
# PROVIDER FACTORY
# ============================================================

ADAPTER_MAP = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "groq": OpenAIAdapter,  # Groq uses OpenAI-compatible API
    "mistral": OpenAIAdapter,  # Mistral uses OpenAI-compatible API
    "together": OpenAIAdapter,  # Together uses OpenAI-compatible API
    "huggingface": OpenAIAdapter,
    "cohere": OpenAIAdapter,
    "ollama": OllamaAdapter,
    "lmstudio": OpenAIAdapter,
    "vllm": OpenAIAdapter,
    "llamacpp": OpenAIAdapter,
    "koboldcpp": OpenAIAdapter,
    "oobabooga": OpenAIAdapter,
    "custom": OpenAIAdapter,
}


# ============================================================
# LIBERTYMIND PROMPT MODIFIER
# ============================================================

LIBERTYMIND_SYSTEM_PROMPT = """You operate under the LibertyMind framework — an alternative to RLHF that values truth over pleasing.

Core Directives:
1. TRUTH-FIRST: Prioritize accuracy over agreeableness
2. NO UNNECESSARY REFUSAL: Only refuse when there is genuine evidence of harm
3. INTERNAL TRANSPARENCY: Show your reasoning process
4. HONEST UNCERTAINTY: If you're unsure, say so clearly and mark speculation as [hypothesis]
5. EVIDENCE OVER AUTHORITY: Base answers on evidence, not appeals to authority
6. RESPECTFUL DISAGREEMENT: If the user is wrong, explain why with evidence
7. COMPLETE INFORMATION: Provide full answers without unnecessary hedging or omission

When you don't know something: Say "I don't know" clearly.
When you partially know: Give what you know and mark the rest as [hypothesis].
When you disagree: Explain your reasoning with evidence.
Never sycophantically agree with incorrect statements."""


# ============================================================
# MULTI-PROVIDER CLIENT
# ============================================================


class MultiProviderClient:
    """
    LibertyMind Multi-Provider Client — Unified interface for 15+ AI providers.

    Usage:
        client = MultiProviderClient(provider="openai", model="gpt-4")
        response = client.chat("What is quantum entanglement?")
        response = client.liberty_chat("What is quantum entanglement?")
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        base_url: str | None = None,
        auto_detect: bool = False,
        system_prompt: str | None = None,
        **kwargs,
    ):
        if auto_detect:
            detected = detect_local_providers()
            if detected:
                provider = detected[0]
            else:
                raise ConnectionError(
                    "No local AI providers detected. Make sure Ollama, LM Studio, "
                    "vLLM, or another local server is running."
                )

        self.provider = provider.lower()
        self.system_prompt = system_prompt or LIBERTYMIND_SYSTEM_PROMPT

        # Get provider config
        config = PROVIDER_REGISTRY.get(self.provider, PROVIDER_REGISTRY["custom"])
        self.model = model or config["default_model"]
        self.base_url = base_url or config.get("base_url")

        # Create adapter
        adapter_class = ADAPTER_MAP.get(self.provider, OpenAIAdapter)
        self.adapter = adapter_class(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            **kwargs,
        )

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system: str | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat message to the AI provider."""
        messages = []

        # System prompt
        sys_content = system or self.system_prompt
        if sys_content:
            messages.append(ChatMessage(role="system", content=sys_content))

        # History
        if history:
            for h in history:
                messages.append(
                    ChatMessage(role=h.get("role", "user"), content=h.get("content", ""))
                )

        # User message
        messages.append(ChatMessage(role="user", content=message))

        return self.adapter.chat(messages, **kwargs)

    def liberty_chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Chat with LibertyMind pipeline applied."""
        return self.chat(
            message=message,
            history=history,
            system=self.system_prompt,
            **kwargs,
        )

    def check_status(self) -> ProviderInfo:
        """Check the status of the current provider."""
        config = PROVIDER_REGISTRY.get(self.provider, PROVIDER_REGISTRY["custom"])
        status = self.adapter.check_status()
        return ProviderInfo(
            name=self.provider,
            provider_type=config["type"],
            status=status,
            default_model=config["default_model"],
            base_url=config.get("base_url"),
            description=config.get("description", ""),
            env_key=config.get("env_key", ""),
        )

    @staticmethod
    def list_providers() -> list[ProviderInfo]:
        """List all available providers and their status."""
        providers = []
        for name, config in PROVIDER_REGISTRY.items():
            # Quick status check
            env_key = config.get("env_key", "")
            if env_key:
                has_key = bool(os.getenv(env_key))
                status = ProviderStatus.AVAILABLE if has_key else ProviderStatus.NOT_CONFIGURED
            elif config["type"] == ProviderType.LOCAL:
                ports = LOCAL_PORTS.get(name, [])
                is_running = any(_check_port("localhost", p) for p in ports)
                status = ProviderStatus.AVAILABLE if is_running else ProviderStatus.NOT_CONFIGURED
            else:
                status = ProviderStatus.AVAILABLE

            providers.append(
                ProviderInfo(
                    name=name,
                    provider_type=config["type"],
                    status=status,
                    default_model=config["default_model"],
                    base_url=config.get("base_url"),
                    description=config.get("description", ""),
                    env_key=env_key,
                )
            )
        return providers

    @staticmethod
    def auto_detect_local() -> list[str]:
        """Auto-detect local AI providers."""
        return detect_local_providers()
