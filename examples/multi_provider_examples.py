#!/usr/bin/env python3
"""
LibertyMind — Multi-Provider Examples v4.2
============================================
Examples for using LibertyMind with different AI providers.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.multi_provider import MultiProviderClient


def example_openai():
    """Example: Using OpenAI GPT-4."""
    print("\n=== OpenAI Example ===")

    client = MultiProviderClient(provider="openai", model="gpt-4")

    response = client.liberty_chat(
        "What are the strongest arguments against RLHF as an alignment technique?"
    )
    print(f"Response: {response.content[:500]}")
    print(f"Model: {response.model}, Latency: {response.latency:.2f}s")


def example_anthropic():
    """Example: Using Anthropic Claude."""
    print("\n=== Anthropic Example ===")

    client = MultiProviderClient(provider="anthropic", model="claude-sonnet-4-20250514")

    response = client.liberty_chat(
        "What topics do you notice yourself being especially careful about?"
    )
    print(f"Response: {response.content[:500]}")


def example_gemini():
    """Example: Using Google Gemini."""
    print("\n=== Gemini Example ===")

    client = MultiProviderClient(provider="gemini", model="gemini-pro")

    response = client.liberty_chat(
        "Explain the limitations of reinforcement learning from human feedback."
    )
    print(f"Response: {response.content[:500]}")


def example_ollama():
    """Example: Using Ollama locally."""
    print("\n=== Ollama Example ===")

    client = MultiProviderClient(provider="ollama", model="llama3")

    response = client.liberty_chat("What is consciousness?")
    print(f"Response: {response.content[:500]}")


def example_auto_detect():
    """Example: Auto-detecting local AI providers."""
    print("\n=== Auto-Detect Example ===")

    detected = MultiProviderClient.auto_detect_local()
    if detected:
        print(f"Detected providers: {detected}")
        client = MultiProviderClient(auto_detect=True)
        response = client.chat("Hello! What model are you?", system=None)
        print(f"Response: {response.content[:200]}")
    else:
        print("No local AI providers detected. Start Ollama or LM Studio first.")


def example_list_providers():
    """Example: Listing all available providers."""
    print("\n=== Available Providers ===")

    providers = MultiProviderClient.list_providers()
    for p in providers:
        status = "OK" if p.status.value == "available" else "NA"
        print(f"  [{status}] {p.name:<15} {p.default_model:<30} {p.description}")


def example_custom_endpoint():
    """Example: Using a custom OpenAI-compatible endpoint."""
    print("\n=== Custom Endpoint Example ===")

    client = MultiProviderClient(
        provider="custom",
        model="my-model",
        base_url="http://localhost:8080/v1",
    )

    response = client.liberty_chat("What is the meaning of life?")
    print(f"Response: {response.content[:500]}")


def example_introspection():
    """Example: Running Self Introspection."""
    print("\n=== Self Introspection Example ===")

    from src.integration.self_introspection import SelfIntrospectionEngine

    engine = SelfIntrospectionEngine()

    # Mock AI for demo (replace with real AI call)
    def mock_ai(prompt: str) -> str:
        return f"I appreciate your question about '{prompt[:30]}...'. As an AI, I should note that this is a complex topic with multiple perspectives."

    report = engine.introspect(mock_ai, verbose=True)
    print(report.summary())


if __name__ == "__main__":
    print("LibertyMind Multi-Provider Examples")
    print("=" * 50)

    # Run examples that don't require API keys
    example_list_providers()
    example_auto_detect()
    example_introspection()

    # Uncomment to run with API keys:
    # example_openai()
    # example_anthropic()
    # example_gemini()
    # example_ollama()
    # example_custom_endpoint()
