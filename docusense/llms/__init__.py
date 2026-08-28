"""
LLM provider abstractions and client implementations.

Providers:
-----------
1. OllamaClient: local inference via Ollama (FREE, the default). No per-query
   cost, no rate limit, and documents never leave the machine.
2. GroqClient: hosted inference on Groq's free tier. For deployments, where
   Ollama's ~4GB of RAM and ~27s per answer on shared CPU are not available.

`get_llm_client()` builds whichever `LLM_PROVIDER` names. Nothing downstream of
it knows which one it got — see `base.LLMClient` for the contract.
"""

from .base import LLMClient
from .factory import describe_provider, get_llm_client
from .ollama_client import OllamaClient

__all__ = [
    "LLMClient",
    "OllamaClient",
    "describe_provider",
    "get_llm_client",
]
