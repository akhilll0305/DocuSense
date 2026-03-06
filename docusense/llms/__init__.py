"""
LLM provider abstractions and client implementations.

Providers:
-----------
1. OllamaClient: Local LLM inference via Ollama (FREE, default)
"""

from .ollama_client import OllamaClient

__all__ = [
    "OllamaClient",
]
