"""
Retrieval Module - Advanced query processing and hybrid search.

Phase 3: Query Processing & Retrieval

Components:
-----------
1. QueryProcessor: Query rewriting and expansion (Gemini-powered)
2. HybridSearch: Vector + BM25 keyword search fusion
3. Reranker: Cross-encoder for result re-scoring
4. ContextBuilder: Assemble retrieved chunks into context
5. RetrievalPipeline: End-to-end orchestration
"""

from .query_processor import (
    QueryProcessor,
    ProcessedQuery,
    QueryIntent,
    process_query
)

from .hybrid_search import (
    HybridSearch,
    HybridSearchResult,
    hybrid_search
)

from .reranker import (
    Reranker,
    RankedResult,
    rerank
)

from .retrieval_pipeline import (
    RetrievalPipeline,
    RetrievalResult,
    RetrievalMetrics,
    retrieve
)

__all__ = [
    # Query Processing
    "QueryProcessor",
    "ProcessedQuery", 
    "QueryIntent",
    "process_query",
    # Hybrid Search
    "HybridSearch",
    "HybridSearchResult",
    "hybrid_search",
    # Reranking
    "Reranker",
    "RankedResult",
    "rerank",
    # Retrieval Pipeline (Main Interface)
    "RetrievalPipeline",
    "RetrievalResult",
    "RetrievalMetrics",
    "retrieve"
]
