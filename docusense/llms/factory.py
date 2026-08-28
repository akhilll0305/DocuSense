"""
Choosing a generation backend.

One place decides, so the pipeline never names a provider.

Author: DocuSense
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from docusense.config.settings import settings
from docusense.llms.base import LLMClient


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """
    Build the configured generation backend.

    Args:
        provider: Override for `LLM_PROVIDER` ("ollama" or "groq")

    Returns:
        A client satisfying `LLMClient`

    Raises:
        ValueError: on an unknown provider name. Falling back silently would
            mean a deployment that thinks it is using a hosted model while
            quietly trying to reach a local Ollama that is not there.
    """
    name = (provider or settings.llm_provider or "ollama").strip().lower()

    if name == "ollama":
        from docusense.llms.ollama_client import OllamaClient
        return OllamaClient()

    if name == "groq":
        from docusense.llms.groq_client import GroqClient
        return GroqClient()

    raise ValueError(
        f"Unknown LLM_PROVIDER '{name}'. Supported: ollama, groq."
    )


def describe_provider() -> str:
    """A short label for logs and diagnostics."""
    name = (settings.llm_provider or "ollama").strip().lower()
    if name == "groq":
        return f"groq:{settings.groq_model}"
    return f"ollama:{settings.ollama_model}"


__all__ = ["get_llm_client", "describe_provider", "LLMClient"]


logger.debug(f"LLM provider configured: {describe_provider()}")
