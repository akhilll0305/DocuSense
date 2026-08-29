"""
The generation backend is a seam, not a hardcoded client.

Local runs use Ollama; a deployment cannot, because llama3.2:3b needs ~4GB of
RAM and ~27s per answer on shared CPU. These tests cover the swap itself — that
the factory honours the setting, that the hosted client speaks the protocol, and
that a misconfiguration is reported rather than silently falling back to a local
server that is not there.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from docusense.config.settings import settings
from docusense.llms.base import LLMClient
from docusense.llms.factory import describe_provider, get_llm_client
from docusense.llms.groq_client import (
    GroqClient,
    GroqEmptyAnswerError,
    GroqModelUnavailableError,
)


# ==============================================================================
# Factory
# ==============================================================================

class TestProviderSelection:
    def test_defaults_to_ollama(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "ollama")
        from docusense.llms.ollama_client import OllamaClient

        assert isinstance(get_llm_client(), OllamaClient)

    def test_selects_groq(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        assert isinstance(get_llm_client(), GroqClient)

    def test_provider_name_is_case_and_space_insensitive(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "  GROQ ")
        assert isinstance(get_llm_client(), GroqClient)

    def test_unknown_provider_raises(self, monkeypatch):
        """
        Falling back quietly would leave a deployment believing it is using a
        hosted model while trying to reach an Ollama that is not there.
        """
        monkeypatch.setattr(settings, "llm_provider", "gpt-9")
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_llm_client()

    def test_explicit_argument_overrides_the_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "ollama")
        assert isinstance(get_llm_client("groq"), GroqClient)

    def test_describe_provider_names_the_model(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_model", "qwen/qwen3.8-27b")
        assert describe_provider() == "groq:qwen/qwen3.8-27b"


# ==============================================================================
# Protocol conformance
# ==============================================================================

def test_both_clients_satisfy_the_protocol():
    """
    The pipeline depends on this surface. A backend missing one method would
    fail at answer time, not at construction.
    """
    from docusense.llms.ollama_client import OllamaClient

    for client in (OllamaClient(), GroqClient()):
        assert isinstance(client, LLMClient), type(client).__name__
        for method in ("is_available", "generate", "chat", "generate_stream",
                       "get_model_info"):
            assert callable(getattr(client, method)), f"{client}.{method}"


# ==============================================================================
# Groq client behaviour (no network)
# ==============================================================================

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """
    Keep every test in this module off the network.

    `GroqClient(api_key="")` does not mean "no key": the constructor falls back
    to `settings.groq_api_key`, so on a machine with a real key in .env these
    tests quietly called the live API — and two of them failed for that reason
    rather than for the behaviour they describe. Pinning the setting makes the
    intent explicit and the result the same everywhere.
    """
    monkeypatch.setattr(settings, "groq_api_key", "test-key-not-real")


def _completion(text: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": text}}]}
    return response


class TestGroqClient:
    def test_generate_returns_the_message_content(self):
        client = GroqClient(api_key="test-key", max_retries=1)
        with patch("httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value.post.return_value = (
                _completion("  BERT is a transformer.  ")
            )
            assert client.generate("What is BERT?") == "BERT is a transformer."

    def test_system_prompt_becomes_a_system_message(self):
        client = GroqClient(api_key="test-key", max_retries=1)
        with patch("httpx.Client") as ctor:
            post = ctor.return_value.__enter__.return_value.post
            post.return_value = _completion("ok")
            client.generate("question", system_prompt="be terse")

        sent = post.call_args.kwargs["json"]["messages"]
        assert sent[0] == {"role": "system", "content": "be terse"}
        assert sent[1] == {"role": "user", "content": "question"}

    def test_missing_key_is_reported_not_retried(self, monkeypatch):
        """
        A missing key is not a transient failure. Say what to set, and say that
        local generation is the alternative.
        """
        monkeypatch.setattr(settings, "groq_api_key", "")
        client = GroqClient(max_retries=3)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            client.generate("anything")

    def test_is_available_is_false_without_a_key_and_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(settings, "groq_api_key", "")
        assert GroqClient().is_available() is False

    def test_generation_retries_then_gives_up(self):
        client = GroqClient(api_key="test-key", max_retries=2, retry_delay=0)
        with patch("httpx.Client") as ctor:
            post = ctor.return_value.__enter__.return_value.post
            post.side_effect = RuntimeError("upstream exploded")
            with pytest.raises(RuntimeError, match="after 2 attempts"):
                client.generate("question")
        assert post.call_count == 2

    def test_streaming_yields_content_fragments(self):
        client = GroqClient(api_key="test-key")
        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
            '',                                   # keep-alive
            'data: ' + json.dumps({"choices": [{"delta": {"content": " world"}}]}),
            'data: [DONE]',
        ]
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = iter(lines)

        with patch("httpx.Client") as ctor:
            stream = ctor.return_value.__enter__.return_value.stream
            stream.return_value.__enter__.return_value = response
            assert "".join(client.generate_stream("hi")) == "Hello world"

    def test_streaming_skips_frames_it_cannot_parse(self):
        """
        Dropping one unfamiliar frame beats ending an answer mid-sentence.
        """
        client = GroqClient(api_key="test-key")
        lines = [
            'data: {not json',
            'data: ' + json.dumps({"unexpected": "shape"}),
            'data: ' + json.dumps({"choices": [{"delta": {"content": "text"}}]}),
            'data: [DONE]',
        ]
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = iter(lines)

        with patch("httpx.Client") as ctor:
            stream = ctor.return_value.__enter__.return_value.stream
            stream.return_value.__enter__.return_value = response
            assert "".join(client.generate_stream("hi")) == "text"

    def test_errors_do_not_carry_the_api_key(self, monkeypatch):
        """
        Failure text reaches log files and SSE error events. The key must not
        travel with it.
        """
        monkeypatch.setattr(settings, "groq_api_key", "sk-secret-value")
        client = GroqClient(api_key="sk-secret-value", max_retries=1, retry_delay=0)
        with patch("httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value.post.side_effect = RuntimeError(
                "401 for url https://api.groq.com/x?key=sk-secret-value"
            )
            with pytest.raises(RuntimeError) as excinfo:
                client.generate("question")
        assert "sk-secret-value" not in str(excinfo.value)
        assert "***" in str(excinfo.value)

    def test_model_info_reports_configuration_without_the_key(self):
        info = GroqClient(api_key="sk-secret-value").get_model_info()
        assert info["provider"] == "groq"
        assert info["configured"] is True
        assert "sk-secret-value" not in json.dumps(info)


# ==============================================================================
# Model availability
#
# These exist because the mocked tests above all passed while every real answer
# returned HTTP 404: the default GROQ_MODEL had been retired, and the only
# availability check was "does /models answer at all". A mock that never
# describes which models the account has cannot catch that, so these mocks
# describe the model list and the 404 the API actually returns.
# ==============================================================================

def _model_list(*models: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": list(models)}
    return response


def _chat_model(model_id: str, context: int = 131072, **overrides) -> dict:
    entry = {
        "id": model_id,
        "active": True,
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "context_window": context,
    }
    entry.update(overrides)
    return entry


def _http_error(status: int, url: str = "https://api.groq.com/openai/v1/chat/completions"):
    request = httpx.Request("POST", url)
    response = httpx.Response(status, request=request, json={
        "error": {
            "message": "The model `x` does not exist or you do not have access to it.",
            "type": "invalid_request_error",
            "code": "model_not_found",
        }
    })
    return httpx.HTTPStatusError("404 Not Found", request=request, response=response)


class TestModelAvailability:
    def test_available_when_the_configured_model_is_in_the_account_list(self):
        client = GroqClient(api_key="test-key", model="qwen/qwen3.8-27b")
        with patch("httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value.get.return_value = _model_list(
                _chat_model("qwen/qwen3.8-27b"), _chat_model("openai/gpt-oss-120b")
            )
            assert client.is_available() is True

    def test_unavailable_when_the_model_was_retired(self):
        """
        The regression this whole section exists for: a valid key, a reachable
        /models, and a model id the account cannot address. Reporting that as
        available is how a dead default shipped.
        """
        client = GroqClient(api_key="test-key", model="llama-3.3-70b-versatile")
        with patch("httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value.get.return_value = _model_list(
                _chat_model("qwen/qwen3.8-27b")
            )
            assert client.is_available() is False

    def test_is_available_does_not_raise_when_the_api_is_unreachable(self):
        client = GroqClient(api_key="test-key")
        with patch("httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value.get.side_effect = RuntimeError("down")
            assert client.is_available() is False

    def test_usable_models_exclude_what_cannot_generate_an_answer(self):
        """
        The account list is not a list of chat models. Suggesting a speech
        model or a 512-token classifier as a replacement would be worse than
        suggesting nothing.
        """
        client = GroqClient(api_key="test-key")
        models = [
            _chat_model("qwen/qwen3.8-27b"),
            _chat_model("speech", output_modalities=["speech"]),
            _chat_model("whisper", input_modalities=["audio"],
                        output_modalities=["transcription"]),
            _chat_model("tiny-classifier", context=512),
            _chat_model("retired-but-listed", active=False),
        ]
        assert client.usable_chat_models(models) == ["qwen/qwen3.8-27b"]


class TestModelNotFound:
    def test_a_404_names_the_model_and_lists_the_alternatives(self):
        client = GroqClient(api_key="test-key", model="llama-3.3-70b-versatile",
                            max_retries=3, retry_delay=0)
        with patch("httpx.Client") as ctor:
            http = ctor.return_value.__enter__.return_value
            failed = MagicMock()
            failed.raise_for_status.side_effect = _http_error(404)
            http.post.return_value = failed
            http.get.return_value = _model_list(_chat_model("qwen/qwen3.8-27b"))

            with pytest.raises(GroqModelUnavailableError) as excinfo:
                client.generate("What is BERT?")

        message = str(excinfo.value)
        assert "llama-3.3-70b-versatile" in message
        assert "qwen/qwen3.8-27b" in message
        assert "GROQ_MODEL" in message

    def test_a_404_is_not_retried(self):
        """
        A retired model id fails identically every time. Retrying it three
        times with backoff turns a configuration error into a slow one.
        """
        client = GroqClient(api_key="test-key", max_retries=3, retry_delay=0)
        with patch("httpx.Client") as ctor:
            http = ctor.return_value.__enter__.return_value
            failed = MagicMock()
            failed.raise_for_status.side_effect = _http_error(404)
            http.post.return_value = failed
            http.get.return_value = _model_list()

            with pytest.raises(GroqModelUnavailableError):
                client.generate("question")
        assert http.post.call_count == 1

    def test_a_rejected_key_is_not_retried_and_says_what_to_check(self):
        client = GroqClient(api_key="test-key", max_retries=3, retry_delay=0)
        with patch("httpx.Client") as ctor:
            http = ctor.return_value.__enter__.return_value
            failed = MagicMock()
            failed.raise_for_status.side_effect = _http_error(401)
            http.post.return_value = failed

            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                client.generate("question")
        assert http.post.call_count == 1

    def test_streaming_reports_a_retired_model_rather_than_a_status_code(self):
        client = GroqClient(api_key="test-key", model="llama-3.3-70b-versatile")
        with patch("httpx.Client") as ctor:
            http = ctor.return_value.__enter__.return_value
            response = MagicMock()
            response.raise_for_status.side_effect = _http_error(404)
            http.stream.return_value.__enter__.return_value = response
            http.get.return_value = _model_list(_chat_model("qwen/qwen3.8-27b"))

            with pytest.raises(GroqModelUnavailableError):
                list(client.generate_stream("question"))


class TestEmptyAnswer:
    def test_an_empty_answer_is_an_error_not_an_empty_string(self):
        """
        Measured on gpt-oss-20b: a 200 response whose whole token budget went
        to the `reasoning` field, leaving `content` empty. Returning "" puts a
        blank answer in the UI with nothing logged.
        """
        client = GroqClient(api_key="test-key", model="openai/gpt-oss-20b",
                            max_tokens=500, max_retries=3, retry_delay=0)
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "", "reasoning": "thinking " * 200},
            }]
        }
        with patch("httpx.Client") as ctor:
            http = ctor.return_value.__enter__.return_value
            http.post.return_value = response
            with pytest.raises(GroqEmptyAnswerError) as excinfo:
                client.generate("question")

        message = str(excinfo.value)
        assert "500" in message and "reasoning" in message
        # Deterministic at temperature 0: retrying just spends the budget again.
        assert http.post.call_count == 1


class TestTruncationIsReported:
    def test_a_truncated_answer_warns(self):
        """
        A cut-off answer reads as a model that trailed off, not one that ran
        out of budget. The text is still returned — it is real — but the log
        is the only place that can say why it stopped.
        """
        client = GroqClient(api_key="test-key", max_tokens=500, max_retries=1)
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "An answer that stops mid-"},
            }]
        }
        # loguru does not route through pytest's caplog; give it its own sink.
        from loguru import logger

        captured = []
        sink = logger.add(captured.append, level="WARNING")
        try:
            with patch("httpx.Client") as ctor:
                ctor.return_value.__enter__.return_value.post.return_value = response
                out = client.generate("question")
        finally:
            logger.remove(sink)

        assert out == "An answer that stops mid-"
        assert any("cut off" in str(m) for m in captured), captured
