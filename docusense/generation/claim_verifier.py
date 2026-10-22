"""
Claim-level verification: is each sentence of an answer entailed by a source?

WHY THIS EXISTS
---------------
Citation validation (answer_generator.py) checks that a citation *points at*
a retrieved source -- that "(Devlin et al., 2018)" names a paper that is
actually on the table. It cannot tell whether the sentence carrying that
citation says what the source says. A model can cite the right paper for the
wrong number, and every check so far would pass it.

This module asks the question directly. The answer is split into claims
(sentences, citation markers removed), and each claim is scored by a natural
language inference model against windows of the retrieved text: a claim is
*supported* when some window entails it. The answer's faithfulness is the
fraction of its claims that are supported.

THE MODEL
---------
DeBERTa-v3-base fine-tuned on MNLI, FEVER and ANLI (MoritzLaurer's, as the
Xenova ONNX export), int8, run through ONNX Runtime -- no torch, in keeping
with the rest of the deployed stack. FEVER is fact verification against
Wikipedia evidence, which is closer to "does this passage support this claim"
than MNLI alone. How well it separates supported from unsupported claims is
measured by scripts/validate_claim_verifier.py, and is not perfect: a flag
means "no passage was found to entail this", not "this is false", and the UI
says so.

WINDOWS, NOT CHUNKS
-------------------
NLI models are trained on short premises. A 500-token chunk as the premise
dilutes the one sentence that matters, and an 800-token chunk is truncated at
512 tokens, which drops whatever came after. Each source is cut into
overlapping windows of a few sentences and the claim keeps its best window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from docusense.config.settings import settings


# Sentence boundary: terminal punctuation, whitespace, then something that can
# start a sentence. Abbreviations common in papers are protected first.
_ABBREVIATIONS = re.compile(
    r"\b(et al|e\.g|i\.e|cf|vs|Fig|Figs|Eq|Eqs|Sec|Tab|approx|resp|etc)\.",
    re.I,
)
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9*-])")
_CITATION = re.compile(
    r"\s*(\[\d+(?:\s*[,-]\s*\d+)*\]|\[Source \d+\]|\((?:[A-Z][^()]*?,\s*(?:\d{4}|n\.d\.)[^()]*)\))"
)
_MARKDOWN = re.compile(r"^\s*(#+\s*|[-*]\s+|\d+[.)]\s+|>\s*)")
_PLACEHOLDER = "\u0000"


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    entailment: float
    contradiction: float
    source_index: Optional[int] = None
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "supported": self.supported,
            "entailment": round(self.entailment, 4),
            "contradiction": round(self.contradiction, 4),
            "source_index": self.source_index,
            "evidence": self.evidence,
        }


@dataclass
class VerificationResult:
    claims: List[ClaimVerdict] = field(default_factory=list)

    @property
    def faithfulness(self) -> Optional[float]:
        """Fraction of claims some source entails; None for an answer with no claims."""
        if not self.claims:
            return None
        return sum(c.supported for c in self.claims) / len(self.claims)

    @property
    def unsupported(self) -> List[ClaimVerdict]:
        return [c for c in self.claims if not c.supported]

    def to_dict(self) -> Dict[str, Any]:
        f = self.faithfulness
        return {
            "faithfulness": None if f is None else round(f, 4),
            "num_claims": len(self.claims),
            "num_unsupported": len(self.unsupported),
            "claims": [c.to_dict() for c in self.claims],
        }


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences without breaking on 'et al.' or 'Fig. 2'."""
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0)[:-1] + _PLACEHOLDER, text)
    parts = []
    for line in protected.splitlines():
        line = line.strip()
        if line:
            parts.extend(_BOUNDARY.split(line))
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


def extract_claims(answer: str, min_words: int = 4) -> List[str]:
    """
    The checkable sentences of an answer.

    Citation markers are removed (they are checked elsewhere, and an NLI model
    reads "(Smith, 2020)" as content). Headings, a trailing sources list, and
    fragments too short to assert anything are skipped.
    """
    claims = []
    for line in answer.splitlines():
        stripped = line.strip()
        if re.match(r"^(\*\*)?(sources|references|citations)(\*\*)?:?\s*$", stripped, re.I):
            break  # everything after is the bibliography
        if stripped.startswith("#"):
            continue
        for sentence in split_sentences(stripped):
            sentence = _MARKDOWN.sub("", sentence)
            sentence = _CITATION.sub("", sentence).replace("**", "").strip()
            if len(sentence.split()) >= min_words:
                claims.append(sentence)
    return claims


def source_windows(text: str, size: int = 4, stride: int = 2) -> List[str]:
    """Overlapping windows of `size` sentences; a short text is one window."""
    sentences = split_sentences(text)
    if len(sentences) <= size:
        return [" ".join(sentences)] if sentences else []
    windows = []
    for start in range(0, len(sentences), stride):
        windows.append(" ".join(sentences[start:start + size]))
        if start + size >= len(sentences):
            break
    return windows


class NLIModel:
    """An ONNX NLI cross-encoder: (premise, hypothesis) -> class probabilities."""

    def __init__(
        self,
        repo: Optional[str] = None,
        model_file: Optional[str] = None,
        max_length: int = 512,
    ):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.repo = repo or settings.claim_verifier_model
        self.model_file = model_file or settings.claim_verifier_file

        tokenizer_path = hf_hub_download(self.repo, "tokenizer.json")
        model_path = hf_hub_download(self.repo, self.model_file)
        config_path = hf_hub_download(self.repo, "config.json")

        # Label order differs between NLI models -- cross-encoder's are
        # (contradiction, entailment, neutral), MoritzLaurer's (entailment,
        # neutral, contradiction) -- and reading the wrong column turns the
        # verifier into its own opposite. So the order is read, not assumed.
        import json
        id2label = json.loads(Path(config_path).read_text(encoding="utf-8"))["id2label"]
        index = {label.lower(): int(i) for i, label in id2label.items()}
        try:
            self.entailment = index["entailment"]
            self.contradiction = index["contradiction"]
        except KeyError:
            raise ValueError(f"{self.repo} has no entailment/contradiction labels: {id2label}")
        self.num_labels = len(index)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=max_length, strategy="only_first")
        self.tokenizer.enable_padding()

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        logger.info(f"✅ NLI model loaded: {self.repo}/{self.model_file}")

    BATCH_SIZE = 16

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """
        (entailment, contradiction) probabilities, shape (len(pairs), 2).

        Returned in that fixed order whatever the model's own label order,
        so callers never index by a model's column numbers.
        """
        if not pairs:
            return np.zeros((0, 2))
        # Shortest premises first, so a batch pads to its own length -- the
        # same trick the reranker uses (see embeddings/backends.py).
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
        out = np.zeros((len(pairs), 2), dtype=np.float32)
        for start in range(0, len(order), self.BATCH_SIZE):
            idx = order[start:start + self.BATCH_SIZE]
            encoded = self.tokenizer.encode_batch([pairs[i] for i in idx])
            feed = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            }
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
            logits = self.session.run(None, feed)[0]
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / probs.sum(axis=1, keepdims=True)
            out[idx] = probs[:, [self.entailment, self.contradiction]]
        return out


class ClaimVerifier:
    """
    Score each claim of an answer against the sources it was generated from.

    Usage:
        verifier = ClaimVerifier()
        result = verifier.verify(answer_text, [chunk.text for chunk in sources])
        result.faithfulness, result.unsupported
    """

    def __init__(self, model: Optional[NLIModel] = None, threshold: Optional[float] = None):
        self._model = model
        self.threshold = threshold if threshold is not None else settings.claim_support_threshold

    @property
    def model(self) -> NLIModel:
        if self._model is None:
            self._model = NLIModel()
        return self._model

    def verify(
        self,
        answer: str,
        sources: Sequence[str],
        claims: Optional[List[str]] = None,
    ) -> VerificationResult:
        claims = extract_claims(answer) if claims is None else claims
        if not claims:
            return VerificationResult()

        windows: List[Tuple[int, str]] = [
            (i, w) for i, text in enumerate(sources) for w in source_windows(text)
        ]
        if not windows:
            return VerificationResult([
                ClaimVerdict(c, False, 0.0, 0.0) for c in claims
            ])

        pairs = [(w, claim) for claim in claims for _, w in windows]
        probs = self.model.predict(pairs).reshape(len(claims), len(windows), 2)

        verdicts = []
        for c, claim in enumerate(claims):
            best = int(probs[c, :, 0].argmax())
            entailment = float(probs[c, best, 0])
            verdicts.append(ClaimVerdict(
                claim=claim,
                supported=entailment >= self.threshold,
                entailment=entailment,
                contradiction=float(probs[c, :, 1].max()),
                source_index=windows[best][0],
                evidence=windows[best][1],
            ))
        return VerificationResult(verdicts)


def verification_of(response: Any) -> Optional[Dict[str, Any]]:
    """
    A response's claim check, if it carries a real one.

    It is persisted as JSON and validated by pydantic on the way out, so
    anything that is not a plain dict -- an older response type, a test
    double -- is reported as no check rather than breaking the answer.
    """
    value = getattr(response, "verification", None)
    return value if isinstance(value, dict) else None


_default: Optional[ClaimVerifier] = None


def get_claim_verifier() -> ClaimVerifier:
    """A process-wide verifier, so the model loads once."""
    global _default
    if _default is None:
        _default = ClaimVerifier()
    return _default


__all__ = [
    "ClaimVerdict",
    "ClaimVerifier",
    "NLIModel",
    "VerificationResult",
    "extract_claims",
    "get_claim_verifier",
    "source_windows",
    "split_sentences",
    "verification_of",
]
