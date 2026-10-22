"""
Training pairs for the embedder, built from QASPER's train and dev splits.

WHAT A PAIR IS
--------------
(question, positive chunk, hard negative chunk). The chunks are the ones the
product would actually store: each paper goes through the real preprocessor
and the real chunker, at the real settings, so the model is trained on the
text distribution it will be asked to embed -- ~500-token Markdown chunks with
a one-sentence overlap, not QASPER's bare paragraphs.

Positives are labelled by `ground_evidence`, the same function that builds the
benchmark's ground truth. Two different matchers would let the fine-tuned
model "improve" partly by agreeing with its own labeller.

HARD NEGATIVES
--------------
In-batch negatives (other questions' positives) mostly come from other papers,
which teaches the model to tell papers apart. The harder problem is telling
apart two chunks *of the same paper*, so each pair also carries a chunk from
the question's own paper that BM25 ranks highly but that is not evidence.

The top BM25 hit is skipped on purpose. QASPER's evidence annotation is not
exhaustive -- annotators cite the paragraph they used, not every paragraph
that would do -- so the single most lexically similar non-evidence chunk is
disproportionately often an unlabelled positive. Training against it would
teach the model to push away correct answers. Rank 2-4 keeps the negatives
hard without leaning on the labels' weakest point.

LEAKAGE
-------
Only `qasper-train-v0.3.json` and `qasper-dev-v0.3.json` are read here. The
benchmark measures on `qasper-test-v0.3.json`, whose papers are disjoint from
both, so a gain on test is a gain on papers the model has never seen.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger

from docusense.evaluation.qasper_harness import ground_evidence
from docusense.evaluation.qasper_loader import QASPERLoader, QASPERPaper

_TOKEN = re.compile(r"\w+")

# Which BM25 ranks (0-based, among non-evidence chunks) a hard negative is
# drawn from. Rank 0 is skipped; see the module docstring.
NEGATIVE_RANKS = (1, 2, 3)


@dataclass
class TrainingPair:
    query: str
    positive: str
    negative: Optional[str]
    paper_id: str


@dataclass
class PaperChunks:
    """One paper chunked exactly as ingestion would chunk it."""
    paper_id: str
    chunks: List[Tuple[str, str]]  # (chunk_id, text)


def chunk_paper(paper: QASPERPaper, preprocessor=None, chunker=None) -> PaperChunks:
    """Reconstruct, preprocess and chunk one paper through the real components."""
    from docusense.ingestion.chunker import SemanticChunker
    from docusense.ingestion.preprocessor import TextPreprocessor

    preprocessor = preprocessor or TextPreprocessor()
    chunker = chunker or SemanticChunker()

    markdown = QASPERLoader.reconstruct_document(paper)
    cleaned = preprocessor.process(markdown).cleaned_text
    chunks = chunker.chunk(cleaned, doc_id=paper.paper_id)
    return PaperChunks(
        paper_id=paper.paper_id,
        chunks=[(f"{paper.paper_id}:{i}", c.text) for i, c in enumerate(chunks)],
    )


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def build_pairs(
    papers: Iterable[QASPERPaper],
    seed: int = 20260827,
    with_negatives: bool = True,
) -> Tuple[List[TrainingPair], Dict[str, PaperChunks], Dict[str, int]]:
    """
    Turn QASPER papers into training pairs.

    Returns (pairs, chunks by paper id, statistics). One pair is emitted per
    (question, relevant chunk), so a question whose evidence spans two chunks
    contributes two positives.
    """
    from rank_bm25 import BM25Okapi

    from docusense.ingestion.chunker import SemanticChunker
    from docusense.ingestion.preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()
    chunker = SemanticChunker()
    rng = random.Random(seed)

    pairs: List[TrainingPair] = []
    by_paper: Dict[str, PaperChunks] = {}
    stats = {
        "papers": 0, "questions": 0, "questions_used": 0,
        "unanswerable": 0, "ungrounded": 0, "pairs": 0, "negatives": 0,
    }

    for paper in papers:
        stats["papers"] += 1
        chunked = chunk_paper(paper, preprocessor, chunker)
        by_paper[paper.paper_id] = chunked
        if not chunked.chunks:
            continue

        text_by_id = dict(chunked.chunks)
        ids = [cid for cid, _ in chunked.chunks]
        bm25 = BM25Okapi([_tokens(t) for _, t in chunked.chunks]) if with_negatives else None

        for question in paper.questions:
            stats["questions"] += 1
            if question.is_unanswerable:
                stats["unanswerable"] += 1
                continue

            evidence = QASPERLoader.body_evidence(question)
            relevant, _, _ = ground_evidence(chunked.chunks, evidence)
            if not relevant:
                stats["ungrounded"] += 1
                continue
            stats["questions_used"] += 1

            negative = None
            if bm25 is not None:
                scores = bm25.get_scores(_tokens(question.question))
                relevant_set = set(relevant)
                ranked = [
                    ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])
                    if ids[i] not in relevant_set
                ]
                candidates = [ranked[r] for r in NEGATIVE_RANKS if r < len(ranked)]
                if candidates:
                    negative = text_by_id[rng.choice(candidates)]
                    stats["negatives"] += 1

            for chunk_id in relevant:
                pairs.append(TrainingPair(
                    query=question.question,
                    positive=text_by_id[chunk_id],
                    negative=negative,
                    paper_id=paper.paper_id,
                ))

    stats["pairs"] = len(pairs)
    logger.info(f"Built {len(pairs)} pairs from {stats['questions_used']} questions")
    return pairs, by_paper, stats


def retrieval_eval_set(
    papers: Iterable[QASPERPaper],
    by_paper: Dict[str, PaperChunks],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, set]]:
    """
    (queries, corpus, relevant docs) in the shape sentence-transformers'
    InformationRetrievalEvaluator takes, pooled across every paper -- the same
    cross-paper setting the benchmark measures.
    """
    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    relevant: Dict[str, set] = {}

    for chunked in by_paper.values():
        corpus.update(dict(chunked.chunks))

    for paper in papers:
        chunked = by_paper.get(paper.paper_id)
        if not chunked:
            continue
        for i, question in enumerate(paper.questions):
            if question.is_unanswerable:
                continue
            ids, _, _ = ground_evidence(
                chunked.chunks, QASPERLoader.body_evidence(question)
            )
            if ids:
                qid = f"{paper.paper_id}:q{i}"
                queries[qid] = question.question
                relevant[qid] = set(ids)

    return queries, corpus, relevant


__all__ = [
    "NEGATIVE_RANKS",
    "PaperChunks",
    "TrainingPair",
    "build_pairs",
    "chunk_paper",
    "retrieval_eval_set",
]
