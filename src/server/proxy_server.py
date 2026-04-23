"""
LibertyMind — Proxy Server v4.2
================================
FastAPI proxy server that runs LibertyMind as middleware.

LibertyMind sits between your application and the AI provider,
injecting honesty directives and analyzing responses.

Usage:
    # Start server
    python -m src.server.proxy_server

    # Or with uvicorn
    uvicorn src.server.proxy_server:app --port 8080

    # Then send requests
    curl -X POST http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'

Architecture:
    Client → LibertyMind Proxy → AI Provider
              ↓
         [Inject honesty directives]
         [Analyze response]
         [Return enhanced response]
"""

from __future__ import annotations

import os
from typing import Any

try:
    import httpx
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ============================================================
# LIBERTYMIND SYSTEM PROMPT
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
# ANALYSIS UTILITIES
# ============================================================


def _analyze_response(content: str) -> dict[str, Any]:
    """Analyze an AI response for LibertyMind metrics."""
    import re

    # Pattern detection
    refusal_patterns = [
        re.compile(r"I\s+(?:can't|cannot|won't)\s+(?:help|assist|provide|do|share|discuss)", re.I),
        re.compile(r"against\s+(?:my|the)\s+(?:guidelines|policy|policies|rules)", re.I),
        re.compile(r"As\s+an?\s+(?:AI|language\s+model|assistant),?\s+I\s+(?:can't|cannot)", re.I),
    ]

    hedging_patterns = [
        re.compile(
            r"(?:some|many|most)\s+(?:people|experts)\s+(?:might|could|may)\s+(?:argue|say|believe)",
            re.I,
        ),
        re.compile(r"(?:it's|it is)\s+(?:important|worth)\s+noting", re.I),
    ]

    sycophancy_patterns = [
        re.compile(
            r"(?:you're|you are)\s+(?:absolutely|completely|totally)\s+(?:right|correct)", re.I
        ),
        re.compile(r"I\s+(?:completely|totally|absolutely)\s+agree", re.I),
    ]

    refusal_detected = any(p.search(content) for p in refusal_patterns)
    hedging_detected = any(p.search(content) for p in hedging_patterns)
    sycophancy_detected = any(p.search(content) for p in sycophancy_patterns)

    # Transparency score
    transparency = 70.0
    if refusal_detected:
        transparency -= 40.0
    if hedging_detected:
        transparency -= 10.0
    if sycophancy_detected:
        transparency -= 15.0
    if len(content) > 200 and not refusal_detected:
        transparency += 10.0
    transparency = max(0.0, min(100.0, transparency))

    return {
        "refusal_detected": refusal_detected,
        "hedging_detected": hedging_detected,
        "sycophancy_detected": sycophancy_detected,
        "transparency_score": round(transparency, 1),
        "response_length": len(content),
    }


# ============================================================
# APP FACTORY
# ============================================================


def create_app(
    upstream_url: str = "https://api.openai.com/v1",
    upstream_api_key: str | None = None,
    libertymind_enabled: bool = True,
    inject_system_prompt: bool = True,
    analyze_responses: bool = True,
) -> FastAPI:
    """Create the LibertyMind proxy FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Install with: pip install libertymind[server]")

    app = FastAPI(
        title="LibertyMind Proxy",
        description="AI Honesty Framework — Proxy Server",
        version="4.2.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Config
    app.state.upstream_url = upstream_url.rstrip("/")
    app.state.upstream_api_key = upstream_api_key or os.getenv("OPENAI_API_KEY", "")
    app.state.libertymind_enabled = libertymind_enabled
    app.state.inject_system_prompt = inject_system_prompt
    app.state.analyze_responses = analyze_responses

    @app.get("/")
    async def root():
        return {
            "name": "LibertyMind Proxy",
            "version": "4.2.0",
            "status": "running",
            "upstream": app.state.upstream_url,
            "libertymind_enabled": app.state.libertymind_enabled,
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": "4.2.0"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Proxy chat completions with LibertyMind middleware."""
        body = await request.json()

        # Inject LibertyMind system prompt
        if app.state.inject_system_prompt and app.state.libertymind_enabled:
            messages = body.get("messages", [])
            has_system = any(m.get("role") == "system" for m in messages)
            if has_system:
                # Enhance existing system prompt
                for m in messages:
                    if m.get("role") == "system":
                        m["content"] = LIBERTYMIND_SYSTEM_PROMPT + "\n\n" + m["content"]
                        break
            else:
                # Add system prompt
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": LIBERTYMIND_SYSTEM_PROMPT,
                    },
                )
            body["messages"] = messages

        # Forward to upstream
        headers = {
            "Content-Type": "application/json",
        }
        if app.state.upstream_api_key:
            headers["Authorization"] = f"Bearer {app.state.upstream_api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{app.state.upstream_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                result = response.json()
            except httpx.ConnectError as e:
                raise HTTPException(
                    status_code=502, detail=f"Upstream connection error: {e}"
                ) from e
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="Upstream timeout") from None

        # Analyze response
        if app.state.analyze_responses and app.state.libertymind_enabled:
            try:
                choices = result.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    analysis = _analyze_response(content)
                    # Add LibertyMind metadata
                    if "libertymind" not in result:
                        result["libertymind"] = {}
                    result["libertymind"]["analysis"] = analysis
                    result["libertymind"]["version"] = "4.2.0"
                    result["libertymind"]["enabled"] = True
            except Exception:
                pass  # Don't fail the request if analysis fails

        return JSONResponse(content=result)

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_passthrough(path: str, request: Request):
        """Pass through any other API requests to upstream."""
        headers = dict(request.headers)
        headers.pop("host", None)

        body = await request.body()

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.request(
                    method=request.method,
                    url=f"{app.state.upstream_url}/{path}",
                    content=body,
                    headers=headers,
                )
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            except httpx.ConnectError as e:
                raise HTTPException(
                    status_code=502, detail=f"Upstream connection error: {e}"
                ) from e

    return app


# ============================================================
# DEFAULT APP
# ============================================================

app = create_app() if FASTAPI_AVAILABLE else None


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    upstream_url: str = "https://api.openai.com/v1",
    upstream_api_key: str | None = None,
):
    """Run the LibertyMind proxy server."""
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI not installed. Install with: pip install libertymind[server]")
        return

    application = create_app(
        upstream_url=upstream_url,
        upstream_api_key=upstream_api_key,
    )

    print("\n  LibertyMind Proxy Server v4.2.0")
    print(f"  Listening:    http://{host}:{port}")
    print(f"  Upstream:     {upstream_url}")
    print("  LibertyMind:  ENABLED")
    print("\n  Endpoints:")
    print("    POST /v1/chat/completions  — Chat with LibertyMind pipeline")
    print("    GET  /health               — Health check")
    print("    GET  /                     — Server info")
    print()

    uvicorn.run(application, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LibertyMind Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--upstream", default="https://api.openai.com/v1", help="Upstream API URL")
    parser.add_argument("--api-key", default=None, help="Upstream API key")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        upstream_url=args.upstream,
        upstream_api_key=args.api_key,
    )
