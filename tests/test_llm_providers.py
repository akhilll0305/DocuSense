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

import pytest

from docusense.config.settings import settings
from docusense.llms.base import LLMClient
from docusense.llms.factory import describe_provider, get_llm_client
from docusense.llms.groq_client import GroqClient


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
        monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
        assert describe_provider() == "groq:llama-3.3-70b-versatile"


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

    def test_missing_key_is_reported_not_retried(self):
        """
        A missing key is not a transient failure. Say what to set, and say that
        local generation is the alternative.
        """
        client = GroqClient(api_key="", max_retries=3)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            client.generate("anything")

    def test_is_available_is_false_without_a_key_and_does_not_raise(self):
        assert GroqClient(api_key="").is_available() is False

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
