"""
Groq generation backend — the hosted option, for deployments.

WHY GROQ
--------
A public instance cannot run Ollama on a free tier: llama3.2:3b wants ~4GB of
RAM and answers in about 27 seconds on a shared CPU. Groq serves open models on
its own inference hardware, has a free tier, and returns a grounded answer in
one to two seconds — which is the difference between a demo someone reads and a
demo someone closes.

The API is OpenAI-compatible, so this is a thin HTTP client over `httpx`, which
is already a dependency. No SDK is added for it.

Local use stays on Ollama. Nothing here is required to run DocuSense.

Author: DocuSense
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Generator, List, Optional

import httpx
from loguru import logger

from docusense.config.settings import settings


class GroqClient:
    """
    Generation through Groq's OpenAI-compatible chat completions API.

    Usage:
        client = GroqClient()
        if client.is_available():
            print(client.generate("What is BERT?"))
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
    ):
        self.model = model or settings.groq_model
        self.api_key = api_key or settings.groq_api_key
        self.base_url = (base_url or settings.groq_base_url).rstrip("/")
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens or settings.answer_max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        logger.info("🤖 GroqClient initialized")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Key present: {bool(self.api_key)}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool = False,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.temperature
            ),
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
        }

    @staticmethod
    def _as_messages(
        prompt: str, system_prompt: Optional[str]
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _scrub(error: Exception) -> str:
        """
        A failure description with no credential in it.

        httpx puts the request URL in its errors, and an upstream body can echo
        headers back. Neither should reach a log file or an SSE error event.
        """
        text = str(error)
        return text.replace(settings.groq_api_key or "\0", "***") if settings.groq_api_key else text

    # ------------------------------------------------------------------
    # LLMClient surface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Whether a key is configured and the model list is reachable."""
        if not self.api_key:
            logger.warning("Groq selected but GROQ_API_KEY is not set")
            return False
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(f"{self.base_url}/models", headers=self._headers())
            if r.status_code == 200:
                logger.info(f"✅ Groq available with model: {self.model}")
                return True
            logger.warning(f"❌ Groq returned {r.status_code} for the model list")
            return False
        except Exception as e:
            logger.warning(f"❌ Groq not reachable: {self._scrub(e)}")
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self.chat(
            self._as_messages(prompt, system_prompt), temperature, max_tokens
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Groq is selected as the LLM provider but GROQ_API_KEY is not "
                "set. Set it, or set LLM_PROVIDER=ollama to generate locally."
            )

        payload = self._payload(messages, temperature, max_tokens)
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
                r.raise_for_status()
                result = r.json()["choices"][0]["message"]["content"].strip()
                logger.success(
                    f"✅ Generated {len(result)} chars in {time.time() - start:.2f}s "
                    f"(attempt {attempt})"
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    f"⚠️ Groq generation failed (attempt {attempt}): {self._scrub(e)}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"Groq generation failed after {self.max_retries} attempts: "
            f"{self._scrub(last_error) if last_error else 'unknown error'}"
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response as server-sent events.

        Not retried: once the first fragment has been yielded the caller has
        already shown it, so a retry would repeat text rather than replace it.
        """
        if not self.api_key:
            raise RuntimeError("Groq is selected but GROQ_API_KEY is not set")

        payload = self._payload(
            self._as_messages(prompt, system_prompt), temperature, None, stream=True
        )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            # A keep-alive or an unfamiliar frame; skipping one
                            # is better than ending an answer mid-sentence.
                            continue
                        content = delta.get("content")
                        if content:
                            yield content
        except Exception as e:
            logger.error(f"❌ Groq stream failed: {self._scrub(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "groq",
            "model": self.model,
            "base_url": self.base_url,
            "configured": bool(self.api_key),
        }
