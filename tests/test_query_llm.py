"""
The query path's LLM is a seam, and most of it is deliberately switched off.

Query rewriting shipped wired to Gemini alone. The key on this project returns
403, so every benchmark run measured the feature as *absent* rather than as
ineffective — a number that looked like "no effect" and actually meant "never
ran". These tests cover the seam that fixed that, and the smaller finding
underneath it: of the three LLM calls the query processor could make, only
rewriting reaches the search at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from docusense.config.settings import settings
from docusense.retrieval.query_processor import QueryProcessor


@pytest.fixture
def llm():
    """A stub generation backend that records what it was asked."""
    client = MagicMock()
    client.model = "stub-model"
    client.generate.return_value = "What accuracy did the model achieve on SST-2?"
    return client


@pytest.fixture(autouse=True)
def _default_flags(monkeypatch):
    """
    Pin the flags these tests are about.

    They are read from .env at import time, so without this the results depend
    on the machine the suite runs on — which is how two Groq tests came to pass
    only where no API key was configured.
    """
    monkeypatch.setattr(settings, "enable_query_rewriting", True)
    monkeypatch.setattr(settings, "enable_query_expansion", False)
    monkeypatch.setattr(settings, "enable_intent_classification", False)


class TestBackendSelection:
    def test_off_makes_no_llm_calls(self, llm):
        qp = QueryProcessor(backend="off", llm_client=llm)
        out = qp.process("What accuracy did they get?")

        assert qp._llm_ready() is False
        llm.generate.assert_not_called()
        # The pattern-based half still runs.
        assert out.rewritten_query == "What accuracy did they get?"
        assert out.metadata["query_llm_backend"] == "off"
        assert out.metadata["query_rewritten"] is False

    def test_provider_backend_rewrites_through_the_given_client(self, llm):
        qp = QueryProcessor(backend="provider", llm_client=llm)
        out = qp.process("What accuracy did they get?")

        assert llm.generate.call_count == 1
        assert out.rewritten_query == "What accuracy did the model achieve on SST-2?"
        assert out.metadata["query_rewritten"] is True
        assert out.metadata["query_llm_available"] is True

    def test_gemini_backend_without_a_key_degrades_rather_than_raising(
        self, monkeypatch
    ):
        monkeypatch.setattr(settings, "gemini_api_key", None)
        qp = QueryProcessor(backend="gemini")
        out = qp.process("What accuracy did they get?")

        assert qp._llm_ready() is False
        assert out.rewritten_query == "What accuracy did they get?"

    def test_unknown_backend_is_treated_as_off(self, llm):
        """
        Unlike LLM_PROVIDER, an unrecognised value here is not fatal: query
        rewriting is an enhancement, and refusing to retrieve at all because
        of a typo in an optional setting would be the worse failure.
        """
        qp = QueryProcessor(backend="something-else", llm_client=llm)
        qp.process("What accuracy did they get?")
        llm.generate.assert_not_called()


class TestOnlyRewritingIsOn:
    def test_expansion_and_intent_are_off_by_default(self, llm):
        """
        Neither reaches retrieval: `retrieval_pipeline` searches on
        `rewritten_query` and reads the other two only to count them. Two LLM
        round trips per query that cannot change a ranking.
        """
        qp = QueryProcessor(backend="provider", llm_client=llm)
        qp.process("What accuracy did they get?")

        assert qp.enable_expansion is False
        assert qp.enable_intent_classification is False
        assert llm.generate.call_count == 1  # rewriting only

    def test_expansion_still_works_when_explicitly_enabled(self, llm, monkeypatch):
        monkeypatch.setattr(settings, "enable_query_expansion", True)
        llm.generate.side_effect = [
            "rewritten question",
            "first variation\nsecond variation",
        ]
        qp = QueryProcessor(backend="provider", llm_client=llm)
        out = qp.process("What accuracy did they get?", num_expansions=2)

        assert llm.generate.call_count == 2
        assert "first variation" in out.expanded_queries

    def test_pattern_expansion_fills_in_when_no_llm_runs(self, llm):
        qp = QueryProcessor(backend="off", llm_client=llm)
        out = qp.process("What accuracy did they get?")
        assert out.expanded_queries  # basic expansion, no API involved


class TestFailureHandling:
    def test_a_failed_rewrite_falls_back_to_the_original_query(self, llm):
        llm.generate.side_effect = RuntimeError("upstream exploded")
        qp = QueryProcessor(backend="provider", llm_client=llm)
        out = qp.process("What accuracy did they get?")

        assert out.rewritten_query == "What accuracy did they get?"
        assert out.metadata["query_rewritten"] is False

    def test_a_permanent_failure_trips_the_breaker_once(self, llm):
        """
        A retired model or a revoked key fails identically forever. Without
        the breaker it logs the same warning on every query for the life of
        the process, and pays the round trip each time.
        """
        llm.generate.side_effect = RuntimeError(
            "Groq model 'x' is not available to this API key"
        )
        qp = QueryProcessor(backend="provider", llm_client=llm)
        qp.process("first question")
        assert qp._llm_ready() is False

        qp.process("second question")
        assert llm.generate.call_count == 1  # not attempted again

    def test_a_transient_failure_does_not_trip_the_breaker(self, llm):
        llm.generate.side_effect = RuntimeError("read timed out")
        qp = QueryProcessor(backend="provider", llm_client=llm)
        qp.process("first question")
        assert qp._llm_ready() is True

        qp.process("second question")
        assert llm.generate.call_count == 2


class TestRewriteAccounting:
    def test_attempts_and_successes_are_counted(self, llm):
        """
        A rewrite that failed and fell back scores exactly like a rewrite that
        did not help. A benchmark of this feature is unreadable without the
        count, so the processor keeps one.
        """
        llm.generate.side_effect = [
            "a good rewrite",
            RuntimeError("read timed out"),
            "another good rewrite",
        ]
        qp = QueryProcessor(backend="provider", llm_client=llm)
        for q in ("one", "two", "three"):
            qp.process(q)

        assert qp.stats == {"rewrites_attempted": 3, "rewrites_succeeded": 2}

    def test_nothing_is_counted_when_the_backend_is_off(self, llm):
        qp = QueryProcessor(backend="off", llm_client=llm)
        qp.process("one")
        assert qp.stats == {"rewrites_attempted": 0, "rewrites_succeeded": 0}


class TestRateLimitBackoff:
    """
    Groq's free tier meters tokens per minute, so a rate limit asks for far
    longer than a linear backoff would guess. Guessing 1s simply burns the
    remaining attempts and reports a failure that was only a wait.
    """

    def _client(self):
        from docusense.llms.groq_client import GroqClient

        return GroqClient(api_key="test-key", retry_delay=1.0, max_retry_wait=65.0)

    def _error(self, status: int, headers: dict | None = None):
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(status, request=request, headers=headers or {})
        return httpx.HTTPStatusError("boom", request=request, response=response)

    def test_a_rate_limit_waits_as_long_as_the_response_asks(self):
        client = self._client()
        assert client._backoff_for(self._error(429, {"retry-after": "42"}), 1) == 42.0

    def test_the_wait_is_capped(self):
        client = self._client()
        assert client._backoff_for(self._error(429, {"retry-after": "9999"}), 1) == 65.0

    def test_a_rate_limit_without_a_header_still_waits_longer_than_linear(self):
        client = self._client()
        linear = 1.0 * 1
        assert client._backoff_for(self._error(429), 1) > linear

    def test_an_unparseable_header_does_not_raise(self):
        client = self._client()
        assert client._backoff_for(self._error(429, {"retry-after": "soon"}), 1) > 0

    def test_other_failures_keep_the_linear_backoff(self):
        client = self._client()
        assert client._backoff_for(RuntimeError("network"), 2) == 2.0
