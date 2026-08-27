"""
RAG Evaluator - End-to-end evaluation orchestrator.

Phase 6: Evaluation & Metrics (Step 3)

Combines retrieval metrics and answer quality metrics into a unified
evaluation framework for the RAG pipeline.

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import time
from pathlib import Path

from loguru import logger

from docusense.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    RetrievalMetricsResult
)
from docusense.evaluation.answer_metrics import (
    AnswerMetrics,
    AnswerMetricsResult
)


@dataclass
class EvaluationResult:
    """Combined evaluation result."""
    retrieval: Optional[RetrievalMetricsResult] = None
    answer: Optional[AnswerMetricsResult] = None
    num_samples: int = 0
    evaluation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "num_samples": self.num_samples,
            "evaluation_time": round(self.evaluation_time, 2),
        }
        if self.retrieval:
            result["retrieval_metrics"] = self.retrieval.to_dict()
        if self.answer:
            result["answer_metrics"] = self.answer.to_dict()
        return result

    def __str__(self) -> str:
        parts = [f"Evaluation Results ({self.num_samples} samples, {self.evaluation_time:.2f}s):"]
        if self.retrieval:
            parts.append(str(self.retrieval))
        if self.answer:
            parts.append(str(self.answer))
        return "\n".join(parts)


@dataclass
class EvaluationSample:
    """A single evaluation sample (ground-truth + predictions)."""
    query: str
    generated_answer: str = ""
    reference_answer: str = ""
    retrieved_ids: List[str] = field(default_factory=list)
    relevant_ids: List[str] = field(default_factory=list)
    source_papers: List[str] = field(default_factory=list)
    relevance_scores: Optional[Dict[str, float]] = None

    # Ground truth as it arrives from a dataset, before ingestion exists.
    # QASPER marks evidence by paragraph *text*; `relevant_ids` can only be
    # filled once those paragraphs have been resolved to real chunk ids.
    evidence_texts: List[str] = field(default_factory=list)
    paper_id: str = ""


class RAGEvaluator:
    """
    End-to-end RAG evaluation.

    Evaluates both retrieval quality and answer quality using:
    - Retrieval: MRR, NDCG, P@K, Recall@K, MAP
    - Answer: ROUGE, citation accuracy, completeness, token overlap

    Usage:
        evaluator = RAGEvaluator()

        samples = [
            EvaluationSample(
                query="What F1 did BERT achieve?",
                generated_answer="BERT achieved 93.5% (Devlin et al., 2018).",
                reference_answer="BERT scored 93.5% F1 on SST-2.",
                retrieved_ids=["chunk_1", "chunk_2", "chunk_3"],
                relevant_ids=["chunk_1", "chunk_3"],
                source_papers=["BERT Paper", "Devlin et al."]
            )
        ]

        result = evaluator.evaluate(samples)
        print(result)
    """

    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerMetrics()
        logger.info("📊 RAGEvaluator initialized")

    def evaluate(
        self,
        samples: List[EvaluationSample],
        evaluate_retrieval: bool = True,
        evaluate_answers: bool = True
    ) -> EvaluationResult:
        """
        Run full evaluation on a set of samples.

        Args:
            samples: List of EvaluationSample objects
            evaluate_retrieval: Compute retrieval metrics
            evaluate_answers: Compute answer metrics

        Returns:
            EvaluationResult with combined metrics
        """
        start = time.time()
        logger.info(f"📊 Evaluating {len(samples)} samples...")

        result = EvaluationResult(num_samples=len(samples))

        # Retrieval evaluation
        if evaluate_retrieval:
            retrieval_evals = [
                {
                    "retrieved": s.retrieved_ids,
                    "relevant": s.relevant_ids,
                    "relevance_scores": s.relevance_scores,
                }
                for s in samples
                if s.retrieved_ids and s.relevant_ids
            ]

            if retrieval_evals:
                result.retrieval = self.retrieval_metrics.evaluate_batch(retrieval_evals)
                logger.info(f"  📚 Retrieval: MRR={result.retrieval.mrr:.4f}, "
                           f"NDCG@5={result.retrieval.ndcg_at_5:.4f}")

        # Answer evaluation
        if evaluate_answers:
            answer_evals = [
                {
                    "generated": s.generated_answer,
                    "reference": s.reference_answer,
                    "query": s.query,
                    "source_papers": s.source_papers,
                }
                for s in samples
                if s.generated_answer
            ]

            if answer_evals:
                result.answer = self.answer_metrics.evaluate_batch(answer_evals)
                logger.info(f"  ✍️ Answer: overlap={result.answer.token_overlap:.4f}, "
                           f"citation_f1={result.answer.citation_f1:.4f}")

        result.evaluation_time = time.time() - start
        logger.success(f"✅ Evaluation complete in {result.evaluation_time:.2f}s")

        return result

    def save_report(
        self,
        result: EvaluationResult,
        output_path: str | Path
    ) -> None:
        """Save evaluation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"📄 Report saved to {output_path}")
