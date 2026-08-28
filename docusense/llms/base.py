"""
The contract every generation backend has to satisfy.

DocuSense generates answers locally through Ollama, which is the right default:
no per-query cost, no rate limit, and documents never leave the machine. It is
the wrong choice for a public instance — llama3.2:3b needs about 4GB of RAM and
takes ~27s per answer on a shared CPU, which is a bad first impression and above
every free hosting tier.

So generation is a seam rather than a hardcoded client. `AnswerGenerator` and
`GenerationPipeline` depend on this surface, not on Ollama.

Author: DocuSense
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """
    What a generation backend must provide.

    Deliberately narrow: the four things the generation pipeline actually
    calls. A backend that cannot stream can implement `generate_stream` by
    yielding one chunk, and the UI still works — it just fills in at once.
    """

    model: str

    def is_available(self) -> bool:
        """
        Whether this backend can serve a request right now.

        Must not raise. A backend that is misconfigured, unreachable, or
        missing its model returns False so the caller can say so plainly
        instead of failing mid-answer.
        """
        ...

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a complete response. Raises RuntimeError on failure."""
        ...

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate from a message list. Raises RuntimeError on failure."""
        ...

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Yield the response in fragments as they arrive."""
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Describe the backend, for /api/health and diagnostics."""
        ...
