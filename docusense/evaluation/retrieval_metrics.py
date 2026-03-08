"""
Retrieval Metrics - MRR, NDCG, Precision@K, Recall@K.

Phase 6: Evaluation & Metrics (Step 1)

PURPOSE:
--------
Quantify retrieval quality with standard IR metrics:
1. MRR (Mean Reciprocal Rank): How high is the first relevant result?
2. NDCG@K (Normalized Discounted Cumulative Gain): Are high-relevance results ranked higher?
3. Precision@K: What fraction of top-K results are relevant?
4. Recall@K: What fraction of all relevant docs are in top-K?
5. MAP (Mean Average Precision): Average precision across recall levels.

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class RetrievalMetricsResult:
    """Aggregated retrieval metrics across multiple queries."""
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    map_score: float = 0.0
    num_queries: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "MRR": round(self.mrr, 4),
            "NDCG@5": round(self.ndcg_at_5, 4),
            "NDCG@10": round(self.ndcg_at_10, 4),
            "P@1": round(self.precision_at_1, 4),
            "P@3": round(self.precision_at_3, 4),
            "P@5": round(self.precision_at_5, 4),
            "P@10": round(self.precision_at_10, 4),
            "Recall@5": round(self.recall_at_5, 4),
            "Recall@10": round(self.recall_at_10, 4),
            "MAP": round(self.map_score, 4),
            "num_queries": self.num_queries,
        }

    def __str__(self) -> str:
        lines = [
            "Retrieval Metrics:",
            f"  MRR:        {self.mrr:.4f}",
            f"  NDCG@5:     {self.ndcg_at_5:.4f}",
            f"  NDCG@10:    {self.ndcg_at_10:.4f}",
            f"  P@1:        {self.precision_at_1:.4f}",
            f"  P@5:        {self.precision_at_5:.4f}",
            f"  Recall@5:   {self.recall_at_5:.4f}",
            f"  Recall@10:  {self.recall_at_10:.4f}",
            f"  MAP:        {self.map_score:.4f}",
            f"  Queries:    {self.num_queries}",
        ]
        return "\n".join(lines)


class RetrievalMetrics:
    """
    Compute standard Information Retrieval metrics.

    Usage:
        metrics = RetrievalMetrics()

        # Single query evaluation
        mrr = metrics.reciprocal_rank(retrieved_ids, relevant_ids)
        ndcg = metrics.ndcg_at_k(retrieved_ids, relevant_ids, k=5)

        # Batch evaluation
        result = metrics.evaluate_batch(queries_with_judgments)
    """

    @staticmethod
    def reciprocal_rank(
        retrieved: List[str],
        relevant: set
    ) -> float:
        """
        Compute Reciprocal Rank (1/rank of first relevant result).

        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs

        Returns:
            Reciprocal rank (0.0 if no relevant found)
        """
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: set,
        k: int
    ) -> float:
        """
        Compute Precision@K.

        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            Fraction of top-K results that are relevant
        """
        if k <= 0 or not retrieved:
            return 0.0

        top_k = retrieved[:k]
        num_relevant = sum(1 for doc in top_k if doc in relevant)
        return num_relevant / k

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: set,
        k: int
    ) -> float:
        """
        Compute Recall@K.

        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            Fraction of relevant docs found in top-K
        """
        if not relevant or k <= 0:
            return 0.0

        top_k = retrieved[:k]
        found = sum(1 for doc in top_k if doc in relevant)
        return found / len(relevant)

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevant: set,
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute NDCG@K (Normalized Discounted Cumulative Gain).

        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff rank
            relevance_scores: Optional per-doc relevance scores (default: binary 1/0)

        Returns:
            NDCG score in [0, 1]
        """
        if not relevant or k <= 0:
            return 0.0

        # Get relevance values for retrieved docs
        def rel_score(doc_id: str) -> float:
            if relevance_scores:
                return relevance_scores.get(doc_id, 0.0)
            return 1.0 if doc_id in relevant else 0.0

        # DCG
        dcg = 0.0
        for i in range(min(k, len(retrieved))):
            rel = rel_score(retrieved[i])
            dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

        # Ideal DCG (sort by relevance)
        if relevance_scores:
            ideal_rels = sorted(
                [relevance_scores.get(d, 0.0) for d in relevant],
                reverse=True
            )
        else:
            ideal_rels = [1.0] * len(relevant)

        idcg = 0.0
        for i in range(min(k, len(ideal_rels))):
            idcg += ideal_rels[i] / math.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def average_precision(
        retrieved: List[str],
        relevant: set
    ) -> float:
        """
        Compute Average Precision for a single query.

        Returns:
            AP score
        """
        if not relevant:
            return 0.0

        hits = 0
        sum_precision = 0.0

        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                hits += 1
                sum_precision += hits / (i + 1)

        return sum_precision / len(relevant)

    def evaluate_batch(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> RetrievalMetricsResult:
        """
        Evaluate retrieval across multiple queries.

        Args:
            evaluations: List of dicts with:
                - "retrieved": List[str] — ordered retrieved doc IDs
                - "relevant": Set[str] — ground-truth relevant doc IDs
                - "relevance_scores" (optional): Dict[str, float]

        Returns:
            RetrievalMetricsResult with averaged metrics
        """
        if not evaluations:
            return RetrievalMetricsResult()

        n = len(evaluations)
        mrr_sum = 0.0
        ndcg5_sum = 0.0
        ndcg10_sum = 0.0
        p1_sum = 0.0
        p3_sum = 0.0
        p5_sum = 0.0
        p10_sum = 0.0
        r5_sum = 0.0
        r10_sum = 0.0
        ap_sum = 0.0

        for ev in evaluations:
            retrieved = ev["retrieved"]
            relevant = set(ev["relevant"])
            scores = ev.get("relevance_scores")

            mrr_sum += self.reciprocal_rank(retrieved, relevant)
            ndcg5_sum += self.ndcg_at_k(retrieved, relevant, 5, scores)
            ndcg10_sum += self.ndcg_at_k(retrieved, relevant, 10, scores)
            p1_sum += self.precision_at_k(retrieved, relevant, 1)
            p3_sum += self.precision_at_k(retrieved, relevant, 3)
            p5_sum += self.precision_at_k(retrieved, relevant, 5)
            p10_sum += self.precision_at_k(retrieved, relevant, 10)
            r5_sum += self.recall_at_k(retrieved, relevant, 5)
            r10_sum += self.recall_at_k(retrieved, relevant, 10)
            ap_sum += self.average_precision(retrieved, relevant)

        return RetrievalMetricsResult(
            mrr=mrr_sum / n,
            ndcg_at_5=ndcg5_sum / n,
            ndcg_at_10=ndcg10_sum / n,
            precision_at_1=p1_sum / n,
            precision_at_3=p3_sum / n,
            precision_at_5=p5_sum / n,
            precision_at_10=p10_sum / n,
            recall_at_5=r5_sum / n,
            recall_at_10=r10_sum / n,
            map_score=ap_sum / n,
            num_queries=n
        )
