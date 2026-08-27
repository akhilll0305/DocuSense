"""
QASPER Harness - run the real retrieval pipeline against the real benchmark.

WHY THIS EXISTS:
----------------
`RAGEvaluator` computes IR metrics from `EvaluationSample` objects, but nothing
ever filled those samples in. `BenchmarkRunner` loaded questions from a dataset
and handed them straight to the evaluator with `retrieved_ids` empty and
`generated_answer` empty, so the evaluator's own filters
(`if s.retrieved_ids and s.relevant_ids`, `if s.generated_answer`) discarded
every sample and produced an empty report in 0.00s. The metrics were correct;
they were simply never given anything to measure.

This module is the missing middle:

    QASPER JSON
        -> reconstruct each paper as Markdown
        -> ingest it through the real DocumentPipeline (chunking, section
           tagging, embeddings, Qdrant)
        -> resolve each question's evidence *paragraphs* to the chunk ids they
           actually landed in                      <- the ground truth
        -> run each retrieval arm over the ingested corpus
        -> hand filled samples to RAGEvaluator

GROUND TRUTH:
-------------
QASPER marks evidence as paragraph text, not as ids, and our chunks are ~500
tokens with overlap, so evidence and chunks do not correspond one to one. A
chunk counts as relevant to an evidence paragraph when it contains that
paragraph verbatim (after whitespace/unicode normalization), or, when the
paragraph straddles a chunk boundary, when the longest contiguous token run
they share covers at least `COVERAGE_THRESHOLD` of either side.

Evidence that cannot be grounded is dropped rather than guessed at, and the
count is reported alongside the metrics: "FLOAT SELECTED: Table 1" entries
point at figures the reconstructed document does not contain, and a small
number of entries name a section rather than quote a paragraph.

ISOLATION:
----------
Everything is ingested under a dedicated `user_id`. Per-user isolation already
scopes SQLite rows, Qdrant filters, and the BM25 corpus, so a benchmark run
cannot see, or be seen by, a real user's documents.

Author: DocuSense
Created: 2026-08-27
"""

from __future__ import annotations

import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from docusense.evaluation.evaluator import (
    EvaluationResult,
    EvaluationSample,
    RAGEvaluator,
)
from docusense.evaluation.qasper_loader import QASPERLoader, QASPERPaper

# A chunk is relevant when the longest contiguous token run it shares with an
# evidence paragraph covers this fraction of either the paragraph or the chunk.
COVERAGE_THRESHOLD = 0.5

# When no chunk clears the threshold, the single best chunk still counts if it
# covers at least this much of the paragraph. Below it, the evidence is treated
# as unmappable and excluded from the ground truth.
FALLBACK_COVERAGE = 0.3

# The benchmark corpus lives under its own tenant.
BENCHMARK_USER_ID = "qasper_benchmark"

_WHITESPACE = re.compile(r"\s+")
_PUNCT_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}


def normalize_text(text: str) -> str:
    """
    Fold text to a form that survives ingestion.

    The preprocessor normalizes unicode and collapses whitespace before
    chunking, so evidence taken from the raw dataset will not match a stored
    chunk byte for byte. Both sides go through this function before comparison.
    """
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _PUNCT_FOLD.items():
        text = text.replace(src, dst)
    return _WHITESPACE.sub(" ", text).strip().casefold()


def coverage(evidence_tokens: Sequence[str], chunk_tokens: Sequence[str]) -> Tuple[float, float]:
    """
    Longest contiguous shared token run, as a fraction of each side.

    Returns (fraction of the evidence covered, fraction of the chunk covered).
    `autojunk` is disabled: on sequences over 200 items SequenceMatcher
    otherwise treats frequent tokens ("the", "of") as junk and refuses to match
    through them, which is exactly the wrong behaviour for prose.
    """
    if not evidence_tokens or not chunk_tokens:
        return 0.0, 0.0

    matcher = SequenceMatcher(None, evidence_tokens, chunk_tokens, autojunk=False)
    match = matcher.find_longest_match(0, len(evidence_tokens), 0, len(chunk_tokens))
    return match.size / len(evidence_tokens), match.size / len(chunk_tokens)


@dataclass
class ArmConfig:
    """One configuration of the retrieval pipeline to measure."""
    name: str
    label: str
    enable_query_processing: bool
    enable_hybrid_search: bool
    enable_reranking: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "query_processing": self.enable_query_processing,
            "hybrid_search": self.enable_hybrid_search,
            "reranking": self.enable_reranking,
        }


# The ablation. Each arm adds exactly one component to the one above it, so a
# difference between two rows is attributable to that component alone.
DEFAULT_ARMS: List[ArmConfig] = [
    ArmConfig(
        name="vector",
        label="Vector only",
        enable_query_processing=False,
        enable_hybrid_search=False,
        enable_reranking=False,
    ),
    ArmConfig(
        name="hybrid",
        label="Hybrid (vector + BM25, RRF)",
        enable_query_processing=False,
        enable_hybrid_search=True,
        enable_reranking=False,
    ),
    ArmConfig(
        name="hybrid_rerank",
        label="Hybrid + cross-encoder rerank",
        enable_query_processing=False,
        enable_hybrid_search=True,
        enable_reranking=True,
    ),
    ArmConfig(
        name="full",
        label="Hybrid + rerank + query processing",
        enable_query_processing=True,
        enable_hybrid_search=True,
        enable_reranking=True,
    ),
]

# Not part of the ladder: what `rag.ask()` did before this benchmark ran.
# `_retrieval_for` built the pipeline in mode="balanced", which forces
# reranking off no matter what `USE_RERANKING` says, so the shipped system
# scored 0.2041 MRR while its own components could reach 0.2777. Kept as an arm
# because that gap is the reason the default changed, and a claim about it
# should stay reproducible.
LEGACY_DEFAULT_ARM = ArmConfig(
    name="legacy_default",
    label="Previous default (balanced, no rerank)",
    enable_query_processing=True,
    enable_hybrid_search=True,
    enable_reranking=False,
)

ALL_ARMS: List[ArmConfig] = DEFAULT_ARMS + [LEGACY_DEFAULT_ARM]


@dataclass
class CorpusStats:
    """What the benchmark corpus actually contains, after ingestion."""
    papers_requested: int = 0
    papers_ingested: int = 0
    chunks: int = 0
    questions_seen: int = 0
    questions_kept: int = 0
    questions_dropped_unanswerable: int = 0
    questions_dropped_no_evidence: int = 0
    evidence_total: int = 0
    evidence_mapped: int = 0
    evidence_unmapped: int = 0
    mean_relevant_chunks: float = 0.0
    ingest_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "papers_requested": self.papers_requested,
            "papers_ingested": self.papers_ingested,
            "chunks": self.chunks,
            "questions_seen": self.questions_seen,
            "questions_kept": self.questions_kept,
            "questions_dropped_unanswerable": self.questions_dropped_unanswerable,
            "questions_dropped_no_evidence": self.questions_dropped_no_evidence,
            "evidence_total": self.evidence_total,
            "evidence_mapped": self.evidence_mapped,
            "evidence_unmapped": self.evidence_unmapped,
            "mean_relevant_chunks": round(self.mean_relevant_chunks, 2),
            "ingest_seconds": round(self.ingest_seconds, 1),
        }


@dataclass
class ArmResult:
    """Metrics for one retrieval arm."""
    arm: ArmConfig
    result: EvaluationResult
    num_queries: int = 0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    empty_results: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    # Per-query scores, in sample order and therefore aligned across arms.
    # An aggregate alone cannot say whether a gap between two arms is real;
    # these make a paired comparison possible, and let any other metric be
    # recomputed without running retrieval again.
    per_query_rr: List[float] = field(default_factory=list)
    per_query_ndcg10: List[float] = field(default_factory=list)

    def to_dict(self, include_per_query: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "arm": self.arm.to_dict(),
            "num_queries": self.num_queries,
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "median_latency_ms": round(self.median_latency_ms, 1),
            "queries_with_no_results": self.empty_results,
        }
        if self.result.retrieval:
            payload["retrieval_metrics"] = self.result.retrieval.to_dict()
        if self.result.answer:
            payload["answer_metrics"] = self.result.answer.to_dict()
        if include_per_query:
            payload["per_query"] = {
                "reciprocal_rank": [round(v, 6) for v in self.per_query_rr],
                "ndcg_at_10": [round(v, 6) for v in self.per_query_ndcg10],
            }
        return payload


@dataclass
class PairedComparison:
    """A paired bootstrap comparison of one arm against another."""
    metric: str
    baseline: str
    variant: str
    baseline_mean: float
    variant_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    num_queries: int

    @property
    def significant(self) -> bool:
        """True when the 95% interval for the difference excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "variant": self.variant,
            "baseline_mean": round(self.baseline_mean, 4),
            "variant_mean": round(self.variant_mean, 4),
            "delta": round(self.delta, 4),
            "ci95_low": round(self.ci_low, 4),
            "ci95_high": round(self.ci_high, 4),
            "p_value": round(self.p_value, 4),
            "significant_at_95": self.significant,
            "num_queries": self.num_queries,
        }


def paired_bootstrap(
    baseline_scores: Sequence[float],
    variant_scores: Sequence[float],
    metric: str,
    baseline_name: str,
    variant_name: str,
    iterations: int = 10000,
    seed: int = 20260827,
) -> PairedComparison:
    """
    Compare two arms on the same queries with a paired bootstrap.

    The arms answer identical questions, so the comparison is paired: resample
    *queries* (carrying both arms' scores together) and recompute the mean
    difference. The 2.5th and 97.5th percentiles give a 95% interval for the
    difference; if it excludes zero, the gap is not attributable to which
    questions happened to be sampled.

    The p-value is a two-sided permutation test: under the null the two arms
    are interchangeable per query, so signs of the per-query differences are
    flipped at random and the observed mean difference is compared against that
    distribution.
    """
    n = min(len(baseline_scores), len(variant_scores))
    if n == 0:
        return PairedComparison(
            metric, baseline_name, variant_name, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0
        )

    baseline_scores = list(baseline_scores[:n])
    variant_scores = list(variant_scores[:n])
    diffs = [v - b for b, v in zip(baseline_scores, variant_scores)]
    observed = sum(diffs) / n

    rng = random.Random(seed)

    boot_means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        boot_means.append(total / n)
    boot_means.sort()

    ci_low = boot_means[int(0.025 * iterations)]
    ci_high = boot_means[min(int(0.975 * iterations), iterations - 1)]

    # Two-sided sign-flip permutation test.
    at_least_as_extreme = 0
    for _ in range(iterations):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= abs(observed):
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (iterations + 1)

    return PairedComparison(
        metric=metric,
        baseline=baseline_name,
        variant=variant_name,
        baseline_mean=sum(baseline_scores) / n,
        variant_mean=sum(variant_scores) / n,
        delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        num_queries=n,
    )


class QASPERHarness:
    """
    Build a QASPER corpus inside DocuSense and measure retrieval against it.

    Usage:
        harness = QASPERHarness(dataset_path="data/benchmarks/qasper/qasper-test-v0.3.json")
        samples, stats = harness.prepare(num_papers=40, seed=20260827)
        results = harness.run_ablation(samples)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        rag: Optional[Any] = None,
        user_id: str = BENCHMARK_USER_ID,
        work_dir: str | Path = "data/benchmarks/qasper/papers",
    ):
        from docusense.rag_pipeline import DocuSenseRAG

        self.dataset_path = Path(dataset_path)
        self.rag = rag or DocuSenseRAG()
        self.user_id = user_id
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.loader = QASPERLoader()
        self.evaluator = RAGEvaluator()

        # paper_id -> document_id, populated by ingestion
        self.document_ids: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Corpus construction
    # ------------------------------------------------------------------

    def select_papers(
        self,
        num_papers: int,
        seed: int = 20260827,
        min_questions: int = 1,
    ) -> List[QASPERPaper]:
        """
        Choose a deterministic sample of papers that carry usable questions.

        Only papers with at least `min_questions` answerable questions backed by
        body-paragraph evidence are eligible, so the sample is not padded with
        papers that contribute nothing to measure.
        """
        papers = self.loader.load(self.dataset_path)

        eligible = [
            p for p in papers
            if sum(
                1 for q in p.questions
                if not q.is_unanswerable and self.loader.body_evidence(q)
            ) >= min_questions
        ]
        eligible.sort(key=lambda p: p.paper_id)  # stable order before sampling

        logger.info(
            f"{len(eligible)}/{len(papers)} papers have >= {min_questions} "
            f"answerable question(s) with body evidence"
        )

        if num_papers >= len(eligible):
            return eligible

        return random.Random(seed).sample(eligible, num_papers)

    def reset_corpus(self) -> int:
        """
        Remove every document previously ingested under the benchmark tenant.

        Scoped to `self.user_id`, so a real user's documents are never touched.
        """
        storage = self.rag.ingestion_pipeline.storage
        docs = storage.get_all_documents(user_id=self.user_id)

        removed = 0
        for doc in docs:
            if self.rag.delete_document(doc.document_id, user_id=self.user_id):
                removed += 1

        if removed:
            logger.info(f"Reset benchmark corpus: removed {removed} document(s)")
        return removed

    def load_existing_document_ids(self) -> Dict[str, str]:
        """
        Recover paper_id -> document_id for a corpus ingested by an earlier run.

        Reconstructed papers are written as `<paper_id>.md`, so the filename
        carries the paper id. Without this, a `--reuse-corpus` run has no
        document ids, cannot look up chunks, and silently grounds no evidence
        at all.
        """
        storage = self.rag.ingestion_pipeline.storage
        for doc in storage.get_all_documents(user_id=self.user_id):
            paper_id = Path(doc.filename).stem
            if paper_id:
                self.document_ids[paper_id] = doc.document_id
        return self.document_ids

    def ingest_papers(
        self,
        papers: List[QASPERPaper],
        skip_existing: bool = False,
    ) -> Tuple[int, float]:
        """
        Ingest reconstructed papers through the real pipeline.

        Args:
            papers: Papers to ingest
            skip_existing: Leave papers already present in the corpus alone

        Returns (papers ingested, seconds elapsed).
        """
        start = time.time()
        ingested = 0

        for i, paper in enumerate(papers, 1):
            if skip_existing and paper.paper_id in self.document_ids:
                continue

            markdown = self.loader.reconstruct_document(paper)
            path = self.work_dir / f"{paper.paper_id}.md"
            path.write_text(markdown, encoding="utf-8")

            result = self.rag.ingest(str(path), user_id=self.user_id)
            if result.success:
                self.document_ids[paper.paper_id] = result.document_id
                ingested += 1
                logger.info(
                    f"[{i}/{len(papers)}] {paper.paper_id}: "
                    f"{result.num_chunks} chunks"
                )
            else:
                logger.warning(f"[{i}/{len(papers)}] {paper.paper_id}: {result.error}")

        elapsed = time.time() - start
        logger.success(f"Ingested {ingested}/{len(papers)} papers in {elapsed:.1f}s")
        return ingested, elapsed

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def _chunks_for(self, paper_id: str) -> List[Tuple[str, str]]:
        """(chunk_id, text) for one ingested paper."""
        document_id = self.document_ids.get(paper_id)
        if not document_id:
            return []
        storage = self.rag.ingestion_pipeline.storage
        return [(c.chunk_id, c.text) for c in storage.get_chunks_by_document(document_id)]

    def resolve_evidence(
        self,
        paper_id: str,
        evidence_texts: Sequence[str],
    ) -> Tuple[List[str], int, int]:
        """
        Map evidence paragraphs onto the chunk ids that contain them.

        Returns (relevant chunk ids, mapped evidence count, unmapped count).
        """
        chunks = self._chunks_for(paper_id)
        if not chunks:
            return [], 0, len(evidence_texts)

        normalized_chunks = [
            (chunk_id, norm, norm.split())
            for chunk_id, norm in (
                (cid, normalize_text(text)) for cid, text in chunks
            )
        ]

        relevant: List[str] = []
        seen = set()
        mapped = 0
        unmapped = 0

        for evidence in evidence_texts:
            evidence_norm = normalize_text(evidence)
            if not evidence_norm:
                unmapped += 1
                continue

            matches = [
                chunk_id for chunk_id, chunk_norm, _ in normalized_chunks
                if evidence_norm in chunk_norm
            ]

            if not matches:
                # The paragraph straddles a chunk boundary, or ingestion
                # reflowed it. Fall back to longest-contiguous-run coverage.
                evidence_tokens = evidence_norm.split()
                best_id, best_evidence_cov = None, 0.0

                for chunk_id, _, chunk_tokens in normalized_chunks:
                    ev_cov, chunk_cov = coverage(evidence_tokens, chunk_tokens)
                    if ev_cov >= COVERAGE_THRESHOLD or chunk_cov >= COVERAGE_THRESHOLD:
                        matches.append(chunk_id)
                    if ev_cov > best_evidence_cov:
                        best_id, best_evidence_cov = chunk_id, ev_cov

                if not matches and best_id and best_evidence_cov >= FALLBACK_COVERAGE:
                    matches = [best_id]

            if matches:
                mapped += 1
                for chunk_id in matches:
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        relevant.append(chunk_id)
            else:
                unmapped += 1

        return relevant, mapped, unmapped

    def prepare(
        self,
        num_papers: int = 40,
        seed: int = 20260827,
        max_questions: Optional[int] = None,
        reset: bool = True,
    ) -> Tuple[List[EvaluationSample], CorpusStats]:
        """
        Build the corpus and the ground truth in one call.

        Returns the samples that are actually measurable, plus the statistics
        describing what was kept and what was dropped.
        """
        stats = CorpusStats(papers_requested=num_papers)

        papers = self.select_papers(num_papers, seed=seed)
        stats.papers_requested = len(papers)

        if reset:
            self.reset_corpus()
        else:
            self.load_existing_document_ids()

        ingested, stats.ingest_seconds = self.ingest_papers(
            papers, skip_existing=not reset
        )
        # On a reuse run the papers already present count towards the corpus
        # even though this call did not ingest them.
        stats.papers_ingested = sum(
            1 for p in papers if p.paper_id in self.document_ids
        ) if not reset else ingested

        storage = self.rag.ingestion_pipeline.storage
        stats.chunks = len(storage.get_all_chunks(user_id=self.user_id))

        samples: List[EvaluationSample] = []
        relevant_counts: List[int] = []

        for paper in papers:
            if paper.paper_id not in self.document_ids:
                continue

            for question in paper.questions:
                stats.questions_seen += 1

                if question.is_unanswerable:
                    stats.questions_dropped_unanswerable += 1
                    continue

                evidence = self.loader.body_evidence(question)
                stats.evidence_total += len(evidence)

                if not evidence:
                    stats.questions_dropped_no_evidence += 1
                    continue

                relevant, mapped, unmapped = self.resolve_evidence(
                    paper.paper_id, evidence
                )
                stats.evidence_mapped += mapped
                stats.evidence_unmapped += unmapped

                if not relevant:
                    stats.questions_dropped_no_evidence += 1
                    continue

                relevant_counts.append(len(relevant))
                samples.append(EvaluationSample(
                    query=question.question,
                    reference_answer=question.answers[0] if question.answers else "",
                    relevant_ids=relevant,
                    evidence_texts=evidence,
                    source_papers=[paper.title] if paper.title else [],
                    paper_id=paper.paper_id,
                ))

                if max_questions and len(samples) >= max_questions:
                    break

            if max_questions and len(samples) >= max_questions:
                break

        stats.questions_kept = len(samples)
        stats.mean_relevant_chunks = (
            sum(relevant_counts) / len(relevant_counts) if relevant_counts else 0.0
        )

        logger.success(
            f"Prepared {stats.questions_kept} measurable questions over "
            f"{stats.papers_ingested} papers ({stats.chunks} chunks)"
        )
        return samples, stats

    # ------------------------------------------------------------------
    # Running the arms
    # ------------------------------------------------------------------

    def _build_pipeline(self, arm: ArmConfig):
        """
        Construct a RetrievalPipeline for one arm.

        `mode="accurate"` is passed because it is the only mode that leaves the
        explicit enable_* flags alone; "fast" and "balanced" overwrite them.
        The arm's own flags decide what actually runs.
        """
        from docusense.retrieval.retrieval_pipeline import RetrievalPipeline

        corpus = self.rag._load_bm25_corpus(self.user_id) if arm.enable_hybrid_search else []

        return RetrievalPipeline(
            vector_store=self.rag.qdrant_store,
            chunks=corpus,
            enable_query_processing=arm.enable_query_processing,
            enable_hybrid_search=arm.enable_hybrid_search,
            enable_reranking=arm.enable_reranking,
            mode="accurate",
        )

    def run_arm(
        self,
        arm: ArmConfig,
        samples: List[EvaluationSample],
        top_k: int = 10,
    ) -> ArmResult:
        """Run every query through one arm and score the rankings."""
        logger.info(f"=== Arm: {arm.label} ({len(samples)} queries) ===")

        pipeline = self._build_pipeline(arm)
        filters = {"user_id": self.user_id}

        scored: List[EvaluationSample] = []
        latencies: List[float] = []
        empty = 0

        for i, sample in enumerate(samples, 1):
            start = time.time()
            try:
                results, _ = pipeline.retrieve(
                    sample.query,
                    top_k=top_k,
                    filters=dict(filters),
                )
            except Exception as exc:  # a failed query is a zero, not a crash
                logger.warning(f"Query {i} failed on arm {arm.name}: {exc}")
                results = []
            latencies.append((time.time() - start) * 1000)

            if not results:
                empty += 1

            scored.append(EvaluationSample(
                query=sample.query,
                reference_answer=sample.reference_answer,
                retrieved_ids=[r.chunk_id for r in results],
                relevant_ids=sample.relevant_ids,
                source_papers=sample.source_papers,
                paper_id=sample.paper_id,
            ))

            if i % 25 == 0 or i == len(samples):
                elapsed = sum(latencies) / 1000
                logger.info(f"  {i}/{len(samples)} queries")
                # Also to stdout: pipeline logs are usually silenced during a
                # run, and a reranking arm can take ten minutes with nothing
                # else to show for it.
                print(
                    f"    {arm.name}: {i}/{len(samples)} queries "
                    f"({elapsed:.0f}s elapsed)",
                    flush=True,
                )

        result = self.evaluator.evaluate(
            scored, evaluate_retrieval=True, evaluate_answers=False
        )

        # Same metrics the aggregate is built from, kept per query so arms can
        # be compared pairwise.
        metrics = self.evaluator.retrieval_metrics
        per_query_rr = [
            metrics.reciprocal_rank(s.retrieved_ids, set(s.relevant_ids))
            for s in scored
        ]
        per_query_ndcg10 = [
            metrics.ndcg_at_k(s.retrieved_ids, set(s.relevant_ids), 10)
            for s in scored
        ]

        latencies_sorted = sorted(latencies)
        median = (
            latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0.0
        )

        return ArmResult(
            arm=arm,
            result=result,
            num_queries=len(samples),
            mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            median_latency_ms=median,
            empty_results=empty,
            latencies_ms=latencies,
            per_query_rr=per_query_rr,
            per_query_ndcg10=per_query_ndcg10,
        )

    def run_ablation(
        self,
        samples: List[EvaluationSample],
        arms: Optional[List[ArmConfig]] = None,
        top_k: int = 10,
    ) -> List[ArmResult]:
        """Run every arm over the same queries and ground truth."""
        arms = arms or DEFAULT_ARMS
        return [self.run_arm(arm, samples, top_k=top_k) for arm in arms]

    # ------------------------------------------------------------------
    # Answer quality
    # ------------------------------------------------------------------

    def evaluate_answers(
        self,
        samples: List[EvaluationSample],
        limit: Optional[int] = None,
        top_k: int = 5,
        seed: int = 20260827,
    ) -> Tuple[EvaluationResult, List[EvaluationSample]]:
        """
        Generate answers with the shipped pipeline and score them.

        This is the end-to-end path (`rag.ask`), so it measures the system as a
        user meets it, local LLM included. Generation is the slow part, so the
        sample is usually smaller than the retrieval sample; the count is
        recorded in the report.
        """
        if limit and limit < len(samples):
            # Samples arrive grouped by paper, so the first N would all come
            # from the same handful of papers. Take a deterministic spread
            # instead, so the answer sample reflects the corpus.
            picked = sorted(random.Random(seed).sample(range(len(samples)), limit))
            subset = [samples[i] for i in picked]
        else:
            subset = samples

        answered: List[EvaluationSample] = []

        for i, sample in enumerate(subset, 1):
            try:
                response = self.rag.ask(
                    sample.query, top_k=top_k, user_id=self.user_id
                )
                generated = getattr(response, "answer", "") or ""
            except Exception as exc:
                logger.warning(f"Generation failed for query {i}: {exc}")
                generated = ""

            answered.append(EvaluationSample(
                query=sample.query,
                generated_answer=generated,
                reference_answer=sample.reference_answer,
                retrieved_ids=sample.retrieved_ids,
                relevant_ids=sample.relevant_ids,
                source_papers=sample.source_papers,
                paper_id=sample.paper_id,
            ))

            logger.info(f"  answered {i}/{len(subset)}")

        result = self.evaluator.evaluate(
            answered, evaluate_retrieval=False, evaluate_answers=True
        )
        return result, answered
