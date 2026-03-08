"""
Evaluation Module - Metrics for RAG system quality.

Phase 6: Evaluation & Metrics

Components:
-----------
1. RetrievalMetrics: MRR, NDCG, Precision@K, Recall@K, MAP
2. AnswerMetrics: ROUGE, citation accuracy, completeness
3. RAGEvaluator: End-to-end evaluation orchestrator
"""

from .retrieval_metrics import (
    RetrievalMetrics,
    RetrievalMetricsResult
)

from .answer_metrics import (
    AnswerMetrics,
    AnswerMetricsResult
)

from .evaluator import (
    RAGEvaluator,
    EvaluationResult,
    EvaluationSample
)

__all__ = [
    "RetrievalMetrics",
    "RetrievalMetricsResult",
    "AnswerMetrics",
    "AnswerMetricsResult",
    "RAGEvaluator",
    "EvaluationResult",
    "EvaluationSample",
]
