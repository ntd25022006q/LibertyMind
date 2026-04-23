"""
LibertyMind - Server Module v4.2
=================================
FastAPI proxy server for running LibertyMind as middleware.

Usage:
    from src.server import create_app

    app = create_app()
    # Then run: uvicorn src.server.proxy_server:app --port 8080
"""

from .proxy_server import create_app

__all__ = [
    "create_app",
]

__version__ = "4.2.0"
