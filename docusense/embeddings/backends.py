"""
Which runtime executes the embedding and reranking models.

WHY THIS EXISTS
---------------
Deployment is memory-bound, and torch is the reason. Measured in one process,
loading and running both models:

    ONNX Runtime (fastembed)     326 MB
    torch / sentence-transformers 758 MB

That difference decides where this can be hosted. At 730MB the whole app needs
a host that will give it a gigabyte, which on a free tier means a credit card
or a subscription; under 512MB the free tiers that ask for neither come into
range.

The swap is free of accuracy consequences because it is the *same models*, not
smaller ones: `sentence-transformers/all-MiniLM-L6-v2` and the ONNX export of
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Verified in one process against the
torch stack — embeddings agree to a cosine of 1.000000, and the cross-encoder
returns byte-identical scores. Retrieval metrics measured under either runtime
are therefore the same metrics, which is the only reason the published numbers
survive this change.

torch stays available (`MODEL_RUNTIME=torch`), because it is what runs on a
GPU, and because a claim that two runtimes agree should be checkable by running
both.

Author: DocuSense
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from loguru import logger

from docusense.config.settings import settings

# The ONNX export of each supported model, and the sequence length its
# sentence-transformers counterpart truncates at. These are not substitutions:
# each entry is the same weights in a different container, which is what lets
# the runtimes be swapped without re-measuring retrieval.
#
# The length is not decoration. fastembed truncates at 128 tokens by default
# while all-MiniLM-L6-v2 under sentence-transformers truncates at 256, and this
# project chunks to ~500 — so on the default settings the two runtimes agree
# perfectly on a short sentence and diverge on every real chunk: measured,
# cosine 0.976 at ~180 tokens and 0.921 at ~600. A silent 0.92 is exactly the
# kind of difference that would have degraded retrieval with nothing to show
# for it, so the length is set explicitly and checked after it is set.
ONNX_EMBEDDING_MODELS: Dict[str, Tuple[str, int]] = {
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", 256),
    "sentence-transformers/all-MiniLM-L6-v2": (
        "sentence-transformers/all-MiniLM-L6-v2", 256
    ),
    "paraphrase-multilingual-MiniLM-L12-v2": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 128
    ),
    "BAAI/bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", 512),
    "BAAI/bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", 512),
}

ONNX_RERANKER_MODELS: Dict[str, str] = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "Xenova/ms-marco-MiniLM-L-12-v2",
    "Xenova/ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base": "BAAI/bge-reranker-base",
}


class UnsupportedModelError(RuntimeError):
    """
    No ONNX build of this model is known.

    Raised rather than falling back to torch, and rather than substituting a
    similar model. A silent substitution would fill the vector store with
    embeddings from a different model than the one the index was built with,
    and nothing downstream would report anything wrong — the results would
    simply get worse.
    """


class EmbeddingBackend(Protocol):
    """The surface `EmbeddingGenerator` depends on."""

    dimension: int

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        ...

    def describe(self) -> Dict[str, Any]:
        ...


class CrossEncoderBackend(Protocol):
    """The surface `Reranker` depends on."""

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        ...

    def describe(self) -> Dict[str, Any]:
        ...


# ==============================================================================
# ONNX Runtime, via fastembed
# ==============================================================================

class OnnxEmbedding:
    def __init__(self, model_name: str, batch_size: int = 32):
        from fastembed import TextEmbedding

        entry = ONNX_EMBEDDING_MODELS.get(model_name)
        if entry is None:
            raise UnsupportedModelError(
                f"No ONNX build is registered for embedding model "
                f"'{model_name}'. Add it to ONNX_EMBEDDING_MODELS if fastembed "
                f"supports it, or set MODEL_RUNTIME=torch."
            )
        repo, max_seq_length = entry
        self.model_name = model_name
        self.repo = repo
        self.max_seq_length = max_seq_length
        self.model = TextEmbedding(repo, batch_size=batch_size)
        self._align_truncation()
        self.dimension = len(next(iter(self.model.embed(["dimension probe"]))))
        logger.info(
            f"✅ ONNX embedding model loaded: {repo} "
            f"({self.dimension} dims, {self.max_seq_length} tokens)"
        )

    def _align_truncation(self) -> None:
        """
        Make this runtime truncate where the torch one truncates.

        fastembed's default is shorter, and the difference is invisible in
        every way except the vectors: no error, no warning, just a quietly
        different embedding for any chunk past the cutoff. Set, then read
        back, because a tokenizer API that moves is the failure this is
        guarding against in the first place.

        Padding has to move with it. fastembed pads every sequence to a fixed
        length equal to its own truncation length, so raising truncation alone
        makes a batch of mixed-length texts produce rows of two different
        widths, and the array they are stacked into is ragged: `ValueError:
        setting an array element with a sequence`, from inside fastembed, on
        any batch holding both a long text and a short one.

        The fix is to pad to the longest item in the batch instead of to a
        fixed width, which is what sentence-transformers does. Padding is
        masked out, so it cannot change a vector — measured, cosine 1.000000
        against both the fixed-width setting and the torch runtime — and not
        doing the arithmetic is worth a lot: a short query embedded in 7.9ms
        rather than 25.8ms, because a 15-token question stops being padded out
        to 256 tokens of work.
        """
        tokenizer = getattr(getattr(self.model, "model", None), "tokenizer", None)
        if tokenizer is None:
            raise UnsupportedModelError(
                f"Cannot reach the tokenizer for '{self.repo}' to set its "
                f"truncation length, so it cannot be confirmed to match the "
                f"torch runtime. Set MODEL_RUNTIME=torch."
            )

        padding = dict(tokenizer.padding or {})
        tokenizer.enable_truncation(max_length=self.max_seq_length)
        if padding:
            # Keep whatever fastembed chose for the pad token and direction;
            # only the width is ours to change.
            padding["length"] = None  # pad to the longest item in the batch
            padding.pop("pad_to_multiple_of", None)
            tokenizer.enable_padding(**padding)

        applied = (tokenizer.truncation or {}).get("max_length")
        if applied != self.max_seq_length:
            raise UnsupportedModelError(
                f"Asked for {self.max_seq_length}-token truncation on "
                f"'{self.repo}' and got {applied!r}. Embeddings would not match "
                f"the torch runtime. Set MODEL_RUNTIME=torch."
            )
        if padding and (tokenizer.padding or {}).get("length") is not None:
            raise UnsupportedModelError(
                f"Padding on '{self.repo}' is still a fixed width. A batch "
                f"holding a long text and a short one would produce ragged "
                f"rows. Set MODEL_RUNTIME=torch."
            )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        vectors = np.array(list(self.model.embed(list(texts), batch_size=batch_size)))
        # fastembed already returns unit vectors; renormalising is a no-op that
        # costs nothing and keeps the contract true if that ever changes.
        if normalize and len(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors

    def describe(self) -> Dict[str, Any]:
        return {
            "runtime": "onnx",
            "model": self.model_name,
            "onnx_repo": self.repo,
            "max_seq_length": self.max_seq_length,
        }


class OnnxCrossEncoder:
    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        repo = ONNX_RERANKER_MODELS.get(model_name)
        if repo is None:
            raise UnsupportedModelError(
                f"No ONNX build is registered for cross-encoder '{model_name}'. "
                f"Add it to ONNX_RERANKER_MODELS if fastembed supports it, or "
                f"set MODEL_RUNTIME=torch."
            )
        self.model_name = model_name
        self.repo = repo
        self.model = TextCrossEncoder(repo)
        logger.info(f"✅ ONNX cross-encoder loaded: {repo}")

    # Small enough that a batch holds documents of similar length once they are
    # sorted, which is the entire point; large enough not to lose the batching.
    BATCH_SIZE = 8

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs, shortest documents first.

        The sort is a performance fix, not a correctness one — scores are
        per-pair and do not depend on what a document is batched with, which
        is checked against the torch runtime to 0.00000. It matters because
        fastembed pads every document in a batch to the longest one in it and
        takes the batch in the order given, so a single batch mixing a
        120-character candidate with a 2,500-character one does the long
        document's work forty times over. sentence-transformers sorts by
        length internally; this is the same trick.

        Measured on 40 candidates of mixed length: 3162ms as given, 1787ms
        sorted — against 2220ms for torch, which is to say the runtime with
        half the memory is also the faster one once it batches sensibly.
        """
        if not pairs:
            return []
        # fastembed scores one query against many documents. Every caller here
        # reranks a single query's results, but grouping keeps that an
        # observation rather than an assumption.
        scores: List[float] = [0.0] * len(pairs)
        by_query: Dict[str, List[int]] = {}
        for i, (query, _) in enumerate(pairs):
            by_query.setdefault(query, []).append(i)

        for query, indices in by_query.items():
            order = sorted(indices, key=lambda i: len(pairs[i][1]))
            documents = [pairs[i][1] for i in order]
            ranked = self.model.rerank(query, documents, batch_size=self.BATCH_SIZE)
            for i, score in zip(order, ranked):
                scores[i] = float(score)
        return scores

    def describe(self) -> Dict[str, Any]:
        return {"runtime": "onnx", "model": self.model_name, "onnx_repo": self.repo}


# ==============================================================================
# torch, via sentence-transformers
# ==============================================================================

class TorchEmbedding:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"✅ torch embedding model loaded: {model_name} ({self.dimension} dims)")

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "runtime": "torch",
            "model": self.model_name,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
        }


class TorchCrossEncoder:
    def __init__(self, model_name: str, max_length: int = 512, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        logger.info(f"✅ torch cross-encoder loaded: {model_name}")

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        return [float(s) for s in self.model.predict(list(pairs))]

    def describe(self) -> Dict[str, Any]:
        return {"runtime": "torch", "model": self.model_name, "device": self.device}


# ==============================================================================
# Selection
# ==============================================================================

def _runtime(override: Optional[str] = None) -> str:
    return (override or settings.model_runtime or "onnx").strip().lower()


def load_embedding_backend(
    model_name: str,
    runtime: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> EmbeddingBackend:
    """
    Build the embedding runtime named by `MODEL_RUNTIME`.

    Raises:
        ValueError: on an unknown runtime name. Falling back quietly would
            mean a deployment that believes it is running the small runtime
            while paying for the large one, which is the whole point of the
            setting.
    """
    name = _runtime(runtime)
    if name == "onnx":
        if device not in ("cpu", ""):
            # onnxruntime's GPU provider is a different package and is not a
            # dependency here. Say so rather than silently running on CPU.
            raise ValueError(
                f"MODEL_RUNTIME=onnx runs on CPU; EMBEDDING_DEVICE={device!r} "
                f"was requested. Set MODEL_RUNTIME=torch for GPU."
            )
        return OnnxEmbedding(model_name, batch_size=batch_size)
    if name == "torch":
        return TorchEmbedding(model_name, device=device)
    raise ValueError(f"Unknown MODEL_RUNTIME '{name}'. Supported: onnx, torch.")


def load_cross_encoder_backend(
    model_name: str,
    runtime: Optional[str] = None,
    max_length: int = 512,
    device: str = "cpu",
) -> CrossEncoderBackend:
    """Build the reranking runtime named by `MODEL_RUNTIME`."""
    name = _runtime(runtime)
    if name == "onnx":
        return OnnxCrossEncoder(model_name)
    if name == "torch":
        return TorchCrossEncoder(model_name, max_length=max_length, device=device)
    raise ValueError(f"Unknown MODEL_RUNTIME '{name}'. Supported: onnx, torch.")


__all__ = [
    "CrossEncoderBackend",
    "EmbeddingBackend",
    "ONNX_EMBEDDING_MODELS",
    "ONNX_RERANKER_MODELS",
    "OnnxCrossEncoder",
    "OnnxEmbedding",
    "TorchCrossEncoder",
    "TorchEmbedding",
    "UnsupportedModelError",
    "load_cross_encoder_backend",
    "load_embedding_backend",
]
