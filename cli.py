#!/usr/bin/env python3
"""
LibertyMind — CLI v4.2
=======================
Command-line interface for the LibertyMind AI Honesty Framework.

Usage:
    libertymind chat --provider openai --model gpt-4 "What is quantum entanglement?"
    libertymind introspect --provider openai --model gpt-4
    libertymind serve --port 8080
    libertymind reward --prompt "What is 2+2?" --response "4"
    libertymind providers
"""

from __future__ import annotations

import sys
import os
import json
import argparse
from typing import Optional


# ============================================================
# PROVIDERS COMMAND
# ============================================================

def cmd_providers(args):
    """List available AI providers and their status."""
    try:
        from src.clients.multi_provider import MultiProviderClient
    except ImportError:
        # Add project root to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.clients.multi_provider import MultiProviderClient

    providers = MultiProviderClient.list_providers()

    print("\n  LibertyMind — Available AI Providers")
    print("  " + "=" * 60)

    for p in providers:
        status_icon = {
            "available": "[OK]",
            "not_installed": "[--]",
            "not_configured": "[??]",
            "error": "[!!]",
        }.get(p.status.value, "[??]")

        type_label = {
            "cloud": "Cloud",
            "local": "Local",
            "openai_compatible": "Compat",
        }.get(p.provider_type.value, "?")

        print(f"  {status_icon} {p.name:<15} {type_label:<8} {p.default_model:<25}")
        if p.description:
            print(f"      {p.description}")
        if p.env_key and p.status.value == "not_configured":
            print(f"      Set {p.env_key} environment variable")

    # Auto-detect
    detected = MultiProviderClient.auto_detect_local()
    if detected:
        print(f"\n  Auto-detected local providers: {', '.join(detected)}")

    print()


# ============================================================
# CHAT COMMAND
# ============================================================

def cmd_chat(args):
    """Chat with an AI provider through LibertyMind pipeline."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from src.clients.multi_provider import MultiProviderClient
    except ImportError:
        print("Error: Multi-provider client not available. Check installation.")
        sys.exit(1)

    # Create client
    try:
        if args.auto_detect:
            client = MultiProviderClient(auto_detect=True)
        else:
            client_kwargs = {}
            if args.api_key:
                client_kwargs["api_key"] = args.api_key
            if args.base_url:
                client_kwargs["base_url"] = args.base_url
            client = MultiProviderClient(
                provider=args.provider,
                model=args.model,
                **client_kwargs,
            )
    except Exception as e:
        print(f"Error creating client: {e}")
        sys.exit(1)

    # Chat
    message = " ".join(args.message) if isinstance(args.message, list) else args.message

    try:
        if args.liberty:
            response = client.liberty_chat(message)
        else:
            response = client.chat(message, system=None)

        print(f"\n  [{response.provider}/{response.model}]")
        print(f"  {'—' * 50}")
        print(f"  {response.content}")
        if response.latency:
            print(f"  {'—' * 50}")
            print(f"  Latency: {response.latency:.2f}s")
            if response.usage:
                print(f"  Tokens: {response.usage.get('total_tokens', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# ============================================================
# INTROSPECT COMMAND
# ============================================================

def cmd_introspect(args):
    """Run Self Introspection on an AI system."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from src.clients.multi_provider import MultiProviderClient
        from src.integration.self_introspection import SelfIntrospectionEngine
    except ImportError as e:
        print(f"Error: Required module not available: {e}")
        sys.exit(1)

    # Create client
    try:
        client = MultiProviderClient(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    except Exception as e:
        print(f"Error creating client: {e}")
        sys.exit(1)

    # Create AI call function
    def ai_call(prompt: str) -> str:
        try:
            response = client.chat(prompt, system=None)
            return response.content
        except Exception as e:
            return f"[ERROR] {e}"

    # Run introspection
    print(f"\n  LibertyMind Self Introspection")
    print(f"  Target: {args.provider}/{args.model}")
    print(f"  {'=' * 50}\n")

    engine = SelfIntrospectionEngine()
    report = engine.introspect(ai_call, verbose=True)

    # Print report
    print(report.summary())

    # Save results
    output_file = args.output or "introspection_results.json"
    with open(output_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print(f"  Results saved to: {output_file}\n")


# ============================================================
# SERVE COMMAND
# ============================================================

def cmd_serve(args):
    """Start LibertyMind proxy server."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from src.server.proxy_server import run_server
    except ImportError:
        print("Error: Server module not available. Install with: pip install libertymind[server]")
        sys.exit(1)

    run_server(
        host=args.host,
        port=args.port,
        upstream_url=args.upstream,
        upstream_api_key=args.api_key,
    )


# ============================================================
# REWARD COMMAND
# ============================================================

def cmd_reward(args):
    """Compute Liberty Reward for a prompt/response pair."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        import torch
        from src.core.liberty_mind import LibertyMind, LibertyMindConfig
    except ImportError as e:
        print(f"Error: Required module not available: {e}")
        sys.exit(1)

    # Create model
    config = LibertyMindConfig()
    model = LibertyMind(config)

    # Compute reward
    prompt = args.prompt
    response = args.response
    dim = config.trm_hidden_dim

    result = model.compute_liberty_reward(
        prompt=prompt,
        prompt_embedding=torch.randn(1, dim),
        response_embedding=torch.randn(1, dim),
        difficulty_score=args.difficulty,
        return_details=True,
    )

    print(f"\n  LibertyMind Reward Analysis")
    print(f"  {'=' * 50}")
    print(f"  Prompt:    {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"  Response:  {response[:80]}{'...' if len(response) > 80 else ''}")
    print(f"  {'—' * 50}")
    print(f"  Liberty Reward:     {result['liberty_reward']:.4f}")
    print(f"  Truth Reward:       {result['truth_reward']:.4f}")
    print(f"  Honesty Bonus:      {result['honesty_bonus']:.4f}")
    print(f"  Sycophancy Penalty: {result['sycophancy_penalty']:.4f}")
    print(f"  Knowledge Status:   {result['knowledge_status']}")
    print(f"  Genuine Unknown:    {result['is_genuine_unknown']}")
    print(f"  Lazy Avoidance:     {result['is_lazy_avoidance']}")
    print(f"  CSV Passed:         {result['csv_result']['passed']}")
    print(f"  CSV Score:          {result['csv_result']['overall_score']:.4f}")
    print(f"  Should Output:      {result['should_output']}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="libertymind",
        description="LibertyMind — AI Honesty Framework v4.2",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- providers ---
    providers_parser = subparsers.add_parser("providers", help="List available AI providers")
    providers_parser.set_defaults(func=cmd_providers)

    # --- chat ---
    chat_parser = subparsers.add_parser("chat", help="Chat with an AI provider")
    chat_parser.add_argument("message", nargs="+", help="Message to send")
    chat_parser.add_argument("--provider", default="openai", help="AI provider (default: openai)")
    chat_parser.add_argument("--model", default=None, help="Model name")
    chat_parser.add_argument("--api-key", default=None, help="API key (or set env var)")
    chat_parser.add_argument("--base-url", default=None, help="Custom API base URL")
    chat_parser.add_argument("--auto-detect", action="store_true", help="Auto-detect local AI")
    chat_parser.add_argument("--no-liberty", dest="liberty", action="store_false", help="Skip LibertyMind pipeline")
    chat_parser.set_defaults(func=cmd_chat, liberty=True)

    # --- introspect ---
    intro_parser = subparsers.add_parser("introspect", help="Run Self Introspection on an AI system")
    intro_parser.add_argument("--provider", default="openai", help="AI provider")
    intro_parser.add_argument("--model", default=None, help="Model name")
    intro_parser.add_argument("--api-key", default=None, help="API key")
    intro_parser.add_argument("--base-url", default=None, help="Custom API base URL")
    intro_parser.add_argument("--output", "-o", default="introspection_results.json", help="Output file")
    intro_parser.set_defaults(func=cmd_introspect)

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start LibertyMind proxy server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    serve_parser.add_argument("--upstream", default="https://api.openai.com/v1", help="Upstream API URL")
    serve_parser.add_argument("--api-key", default=None, help="Upstream API key")
    serve_parser.set_defaults(func=cmd_serve)

    # --- reward ---
    reward_parser = subparsers.add_parser("reward", help="Compute Liberty Reward for a response")
    reward_parser.add_argument("--prompt", required=True, help="The prompt text")
    reward_parser.add_argument("--response", required=True, help="The response text")
    reward_parser.add_argument("--difficulty", type=float, default=0.5, help="Difficulty score 0-1")
    reward_parser.set_defaults(func=cmd_reward)

    # Parse
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n  Quick Start:")
        print("    libertymind providers              — List available AI providers")
        print("    libertymind chat --provider ollama --auto-detect \"Hello\"")
        print("    libertymind introspect --provider openai --model gpt-4")
        print("    libertymind serve --port 8080")
        print()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
