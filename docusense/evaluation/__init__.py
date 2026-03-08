"""
Evaluation Module - Metrics for RAG system quality.

Phase 6: Evaluation & Metrics

Components:
-----------
1. RetrievalMetrics: MRR, NDCG, Precision@K, Recall@K, MAP
2. AnswerMetrics: ROUGE, citation accuracy, completeness
3. RAGEvaluator: End-to-end evaluation orchestrator
4. QASPERLoader: QASPER benchmark dataset loader
5. BenchmarkRunner: Benchmark execution and reporting
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

from .qasper_loader import (
    QASPERLoader,
    QASPERQuestion,
    QASPERPaper
)

from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkReport
)

__all__ = [
    "RetrievalMetrics",
    "RetrievalMetricsResult",
    "AnswerMetrics",
    "AnswerMetricsResult",
    "RAGEvaluator",
    "EvaluationResult",
    "EvaluationSample",
    "QASPERLoader",
    "QASPERQuestion",
    "QASPERPaper",
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkReport",
]

