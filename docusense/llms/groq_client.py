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


class GroqModelUnavailableError(RuntimeError):
    """
    The configured model id is not one this key can use.

    Its own type because it is not a transient failure and must not be retried:
    a retired model id 404s identically three times, a second apart, and then
    reports a network-shaped error for a configuration problem.
    """


class GroqEmptyAnswerError(RuntimeError):
    """
    The API answered, and the answer was empty.

    Reasoning models (the gpt-oss family) return their chain of thought in a
    separate `reasoning` field that is charged against `max_tokens`. A budget
    small enough to be spent entirely on reasoning yields `finish_reason:
    length` with `content: ""` — a 200 response carrying nothing to show.
    """


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

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Every model this key can address, as the API describes them.

        Raises rather than returning [] on failure: an empty list and an
        unreachable API are different facts, and the callers below only use
        this to explain a failure that has already happened.
        """
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.base_url}/models", headers=self._headers())
        r.raise_for_status()
        return r.json().get("data", [])

    def usable_chat_models(
        self, models: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        The subset of the account's models that could generate an answer here.

        The account list is not a list of chat models — it also carries speech
        synthesis, transcription and prompt classifiers, and offering those as
        alternatives would be worse than offering none. Filtered on what the
        API states rather than on a hardcoded list of names: text in, text out,
        active, and a context window that can hold this deployment's prompt.
        """
        needed = settings.max_context_tokens + settings.answer_max_tokens
        try:
            models = models if models is not None else self.list_models()
        except Exception as e:
            logger.debug(f"Could not list Groq models: {self._scrub(e)}")
            return []

        usable = []
        for m in models:
            if not m.get("active", True):
                continue
            if "text" not in (m.get("input_modalities") or ["text"]):
                continue
            if "text" not in (m.get("output_modalities") or ["text"]):
                continue
            if (m.get("context_window") or 0) < needed:
                continue
            usable.append(m["id"])
        return sorted(usable)

    def _unavailable_model_error(self) -> GroqModelUnavailableError:
        """
        Say which model is missing and what the account can use instead.

        Groq answers a retired id with a plain 404, which reaches a log or an
        SSE error event as `Client error '404 Not Found'` — true, and useless
        for the one thing that fixes it.
        """
        alternatives = self.usable_chat_models()
        if alternatives:
            options = "\n  ".join(alternatives)
            suggestion = (
                f"\nModels this key can use for generation:\n  {options}"
                f"\nSet GROQ_MODEL to one of them."
            )
        else:
            suggestion = (
                "\nThe model list could not be read, so there is nothing to "
                "suggest. Check the key at https://console.groq.com."
            )
        return GroqModelUnavailableError(
            f"Groq model '{self.model}' is not available to this API key — it "
            f"has most likely been retired.{suggestion}"
        )

    def _empty_answer_error(self, choice: Dict[str, Any]) -> GroqEmptyAnswerError:
        """
        Explain a 200 that carried no answer.

        Measured on this account: openai/gpt-oss-20b, asked a comparison
        question with max_tokens=500, spent all 500 tokens on `reasoning` and
        returned an empty `content`. Returning "" from here would put a blank
        answer in the UI with nothing logged.
        """
        reason = choice.get("finish_reason") or "unknown"
        reasoning = (choice.get("message", {}).get("reasoning") or "").strip()
        if reasoning and reason == "length":
            cause = (
                f" The model spent its whole {self.max_tokens}-token budget on "
                f"an internal reasoning trace ({len(reasoning)} chars) and left "
                f"nothing for the answer. Raise ANSWER_MAX_TOKENS, or use a "
                f"model that does not reason separately."
            )
        elif reason == "length":
            cause = f" It stopped at the {self.max_tokens}-token limit."
        else:
            cause = ""
        return GroqEmptyAnswerError(
            f"Groq model '{self.model}' returned an empty answer "
            f"(finish_reason: {reason}).{cause}"
        )

    @staticmethod
    def _is_model_not_found(error: Exception) -> bool:
        """Whether an httpx error is Groq's 'this model does not exist'."""
        if not isinstance(error, httpx.HTTPStatusError):
            return False
        return error.response.status_code == 404

    @staticmethod
    def _is_auth_failure(error: Exception) -> bool:
        return (
            isinstance(error, httpx.HTTPStatusError)
            and error.response.status_code in (401, 403)
        )

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
        """
        Whether a key is configured *and* the configured model exists.

        Checking only that /models answers is what made a dead default look
        healthy: the key was valid, the endpoint returned 200, and every
        /chat/completions call 404'd because the model had been retired. A
        reachable API with an unusable model is not availability.
        """
        if not self.api_key:
            logger.warning("Groq selected but GROQ_API_KEY is not set")
            return False
        try:
            models = self.list_models()
        except Exception as e:
            logger.warning(f"❌ Groq not reachable: {self._scrub(e)}")
            return False

        if any(m.get("id") == self.model for m in models):
            logger.info(f"✅ Groq available with model: {self.model}")
            return True

        alternatives = self.usable_chat_models(models)
        logger.warning(
            f"❌ Groq model '{self.model}' is not available to this key. "
            f"Usable for generation: {', '.join(alternatives) or 'none found'}"
        )
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
                choice = r.json()["choices"][0]
                result = (choice["message"].get("content") or "").strip()
                if not result:
                    raise self._empty_answer_error(choice)
                logger.success(
                    f"✅ Generated {len(result)} chars in {time.time() - start:.2f}s "
                    f"(attempt {attempt})"
                )
                return result
            except (GroqModelUnavailableError, GroqEmptyAnswerError):
                raise
            except Exception as e:
                # A model that does not exist and a key that is rejected will
                # fail identically on every attempt. Retrying them turns a
                # legible configuration error into a slow, network-shaped one.
                if self._is_model_not_found(e):
                    raise self._unavailable_model_error() from e
                if self._is_auth_failure(e):
                    raise RuntimeError(
                        f"Groq rejected the API key ({self._scrub(e)}). Check "
                        f"GROQ_API_KEY, or set LLM_PROVIDER=ollama to generate "
                        f"locally."
                    ) from e
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
        except httpx.HTTPStatusError as e:
            # A streamed 404 reaches the browser as an SSE error event, which
            # is the one place a bare status code helps nobody.
            if self._is_model_not_found(e):
                logger.error(f"❌ Groq stream failed: model '{self.model}' unavailable")
                raise self._unavailable_model_error() from e
            logger.error(f"❌ Groq stream failed: {self._scrub(e)}")
            raise
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
