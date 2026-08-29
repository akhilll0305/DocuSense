"""
The two model runtimes have to agree, or the published numbers are for a system
that no longer exists.

Retrieval was measured on QASPER with the torch models. Running the ONNX
exports instead is only safe because they are the *same weights* — so this is
the test that has to hold, and it is the one that nearly did not: fastembed
truncates at 128 tokens by default while all-MiniLM-L6-v2 under
sentence-transformers truncates at 256, and this project chunks to ~500. On the
default settings the two agreed perfectly on a short sentence and diverged on
every real chunk (cosine 0.976 at ~180 tokens, 0.921 at ~600) with nothing
raised and nothing logged.

The cross-runtime tests need torch, which is not a default dependency, and skip
without it:

    pip install -r requirements-torch.txt
    python -m pytest tests/test_model_runtime.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from docusense.config.settings import settings
from docusense.embeddings.backends import (
    ONNX_EMBEDDING_MODELS,
    UnsupportedModelError,
    load_cross_encoder_backend,
    load_embedding_backend,
)

TORCH_INSTALLED = importlib.util.find_spec("sentence_transformers") is not None
needs_torch = pytest.mark.skipif(
    not TORCH_INSTALLED,
    reason="torch runtime not installed (pip install -r requirements-torch.txt)",
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SHORT = "The queue-aware controller reduced average vehicle delay by 19.7%."
# Long enough to cross both truncation limits, which is the whole point.
LONG = (
    "Recent work reports large improvements from learned signal controllers, "
    "usually measured against a fixed-time plan. Fixed-time plans are a weak "
    "baseline: they are computed once and never adapt. "
) * 12


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestSelection:
    def test_default_runtime_is_onnx(self):
        """
        The default decides where this can be hosted: 326MB against 758MB for
        the same two models.
        """
        assert settings.model_runtime == "onnx"

    def test_unknown_runtime_raises(self):
        with pytest.raises(ValueError, match="Unknown MODEL_RUNTIME"):
            load_embedding_backend(EMBEDDING_MODEL, runtime="jax")

    def test_a_model_with_no_registered_onnx_build_raises(self):
        """
        Rather than substituting a similar model. A substitution would fill the
        vector store with embeddings from a different model than the index was
        built with, and nothing downstream would report anything wrong.
        """
        with pytest.raises(UnsupportedModelError, match="No ONNX build"):
            load_embedding_backend("some-model-nobody-exported", runtime="onnx")

    def test_onnx_refuses_a_gpu_request_rather_than_silently_using_cpu(self):
        with pytest.raises(ValueError, match="runs on CPU"):
            load_embedding_backend(EMBEDDING_MODEL, runtime="onnx", device="cuda")


class TestOnnxRuntime:
    def test_truncation_is_set_to_the_torch_length(self):
        """
        Not fastembed's default. This is the assertion that stands between the
        two runtimes agreeing and quietly disagreeing on every real chunk.
        """
        backend = load_embedding_backend(EMBEDDING_MODEL, runtime="onnx")
        expected = ONNX_EMBEDDING_MODELS[EMBEDDING_MODEL][1]
        assert backend.max_seq_length == expected == 256

        tokenizer = backend.model.model.tokenizer
        assert tokenizer.truncation["max_length"] == expected

    def test_padding_is_dynamic(self):
        """
        Fixed-width padding is what made a mixed-length batch ragged, and it
        also pads a 15-token question out to 256 tokens of arithmetic.
        """
        backend = load_embedding_backend(EMBEDDING_MODEL, runtime="onnx")
        assert backend.model.model.tokenizer.padding["length"] is None

    def test_a_batch_of_mixed_lengths_embeds(self):
        """The case that raised ValueError from inside fastembed."""
        backend = load_embedding_backend(EMBEDDING_MODEL, runtime="onnx")
        vectors = backend.encode([LONG, "short", SHORT, LONG])
        assert vectors.shape == (4, settings.embedding_dimension)

    def test_vectors_are_unit_length(self):
        backend = load_embedding_backend(EMBEDDING_MODEL, runtime="onnx")
        vectors = backend.encode([SHORT, LONG], normalize=True)
        for v in vectors:
            assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

    def test_describe_reports_the_runtime_and_the_length(self):
        backend = load_embedding_backend(EMBEDDING_MODEL, runtime="onnx")
        described = backend.describe()
        assert described["runtime"] == "onnx"
        assert described["max_seq_length"] == 256


@needs_torch
class TestRuntimesAgree:
    def test_embeddings_match_on_a_short_text(self):
        onnx = load_embedding_backend(EMBEDDING_MODEL, runtime="onnx")
        torch = load_embedding_backend(EMBEDDING_MODEL, runtime="torch")
        assert cosine(onnx.encode([SHORT])[0], torch.encode([SHORT])[0]) > 0.9999

    def test_embeddings_match_on_a_text_longer_than_the_default_truncation(self):
        """
        The regression this file exists for. With fastembed's default 128-token
        truncation this cosine was 0.921.
        """
        onnx = load_embedding_backend(EMBEDDING_MODEL, runtime="onnx")
        torch = load_embedding_backend(EMBEDDING_MODEL, runtime="torch")
        assert cosine(onnx.encode([LONG])[0], torch.encode([LONG])[0]) > 0.9999

    def test_both_runtimes_report_the_same_dimension(self):
        onnx = load_embedding_backend(EMBEDDING_MODEL, runtime="onnx")
        torch = load_embedding_backend(EMBEDDING_MODEL, runtime="torch")
        assert onnx.dimension == torch.dimension == settings.embedding_dimension

    def test_cross_encoder_scores_match(self):
        onnx = load_cross_encoder_backend(RERANKER_MODEL, runtime="onnx")
        torch = load_cross_encoder_backend(RERANKER_MODEL, runtime="torch")
        pairs = [
            ("What delay reduction was reported?", SHORT),
            ("What delay reduction was reported?", LONG),
            ("Which languages are covered?", SHORT),
        ]
        onnx_scores = onnx.predict(pairs)
        torch_scores = torch.predict(pairs)
        assert len(onnx_scores) == len(torch_scores) == 3
        for a, b in zip(onnx_scores, torch_scores):
            assert abs(a - b) < 1e-3

    def test_cross_encoder_ranks_identically(self):
        """
        Scores matter less than the order they produce, which is what
        reranking actually contributes.
        """
        onnx = load_cross_encoder_backend(RERANKER_MODEL, runtime="onnx")
        torch = load_cross_encoder_backend(RERANKER_MODEL, runtime="torch")
        query = "What delay reduction did the queue-aware controller report?"
        docs = [LONG, SHORT, "Unrelated text about subword vocabularies."]
        pairs = [(query, d) for d in docs]
        assert np.argsort(onnx.predict(pairs)).tolist() == (
            np.argsort(torch.predict(pairs)).tolist()
        )


class TestRerankBatching:
    def test_scores_do_not_depend_on_input_order(self):
        """
        `predict` reorders its input by document length before scoring, to stop
        one long candidate from padding every short one up to its size. That is
        only safe because a pair's score does not depend on its neighbours —
        so the same pairs, shuffled, must come back with the same scores.
        """
        backend = load_cross_encoder_backend(RERANKER_MODEL, runtime="onnx")
        query = "What delay reduction was reported?"
        docs = [SHORT, LONG, "tiny", LONG[:400], SHORT * 3]
        forward = backend.predict([(query, d) for d in docs])
        backward = backend.predict([(query, d) for d in reversed(docs)])
        for a, b in zip(forward, list(reversed(backward))):
            assert abs(a - b) < 1e-6

    def test_several_queries_in_one_call_are_scored_against_their_own_query(self):
        backend = load_cross_encoder_backend(RERANKER_MODEL, runtime="onnx")
        mixed = [
            ("What delay reduction was reported?", SHORT),
            ("Which languages are covered?", SHORT),
            ("What delay reduction was reported?", LONG),
        ]
        together = backend.predict(mixed)
        apart = [backend.predict([pair])[0] for pair in mixed]
        for a, b in zip(together, apart):
            assert abs(a - b) < 1e-6


class TestEmptyInput:
    def test_cross_encoder_handles_no_pairs(self):
        backend = load_cross_encoder_backend(RERANKER_MODEL, runtime="onnx")
        assert backend.predict([]) == []

    def test_embedding_handles_a_single_text(self):
        backend = load_embedding_backend(EMBEDDING_MODEL, runtime="onnx")
        vectors = backend.encode([SHORT])
        assert vectors.shape == (1, settings.embedding_dimension)
