"""
LibertyMind - Clients Module v0.1.0
==================================
Multi-provider AI client adapters.

Supports 15+ AI providers with a unified interface:
- Cloud: OpenAI, Anthropic, Gemini, Groq, Mistral, Together, HuggingFace, Cohere
- Local: Ollama, LM Studio, vLLM, llama.cpp, KoboldCPP, Oobabooga
- Any OpenAI-compatible endpoint
"""

from .multi_provider import MultiProviderClient

__all__ = [
    "MultiProviderClient",
]

__version__ = "0.1.0"
