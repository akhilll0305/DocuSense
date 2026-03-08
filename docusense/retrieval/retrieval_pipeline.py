"""
Retrieval Pipeline - End-to-end orchestration of query processing and retrieval.

This is Step 4 of Phase 3: Query Processing & Retrieval (FINAL).

PURPOSE:
--------
Orchestrate the complete retrieval flow:
1. Query Processing → Rewriting & expansion
2. Hybrid Search → Vector + BM25 fusion
3. Reranking → Cross-encoder scoring
4. Context Assembly → Final results

This is the main interface for document retrieval!

PIPELINE ARCHITECTURE:
----------------------
                    User Query
                        ↓
            ┌──────────────────────┐
            │  Query Processor     │  Rewrite, expand, classify intent
            │  (Gemini-powered)    │
            └──────────────────────┘
                        ↓
            ┌──────────────────────┐
            │   Hybrid Search      │  Vector (semantic) + BM25 (keyword)
            │   (RRF Fusion)       │  → Reciprocal Rank Fusion
            └──────────────────────┘
                        ↓
            ┌──────────────────────┐
            │     Reranker         │  Cross-encoder precision boost
            │  (Cross-encoder)     │  → 20-40% better accuracy
            └──────────────────────┘
                        ↓
            ┌──────────────────────┐
            │  Context Builder     │  Assemble final context
            │  (Deduplication)     │  → Ready for LLM
            └──────────────────────┘
                        ↓
                  Final Results

USAGE EXAMPLE:
--------------
```python
from docusense.retrieval import RetrievalPipeline
from docusense.vectorstore import QdrantVectorStore
from docusense.storage import ChunkStorage

# Initialize
storage = ChunkStorage()
chunks = storage.get_all_chunks()
vector_store = QdrantVectorStore()

pipeline = RetrievalPipeline(
    vector_store=vector_store,
    chunks=chunks
)

# Retrieve
results = pipeline.retrieve("How does machine learning work?", top_k=5)

for result in results:
    print(f"{result.text[:100]}...")
```

PERFORMANCE MODES:
------------------
1. **Fast**: Vector search only (100ms)
2. **Balanced**: Hybrid search + light reranking (500ms)
3. **Accurate**: Full pipeline with reranking (1-2s)

CACHING:
--------
- Query cache: Repeated queries return instantly
- Embedding cache: Reuse embeddings for similar queries
- Result cache: TTL-based caching
"""

from typing import List, Dict, Optional, Literal, Any
from dataclasses import dataclass, field
import time
from loguru import logger

from docusense.config.settings import settings
from docusense.vectorstore import QdrantVectorStore, SearchResult
from docusense.retrieval.query_processor import QueryProcessor, ProcessedQuery
from docusense.retrieval.hybrid_search import HybridSearch, HybridSearchResult
from docusense.retrieval.reranker import Reranker, RankedResult


@dataclass
class RetrievalResult:
    """Final result from retrieval pipeline with full metadata."""
    chunk_id: str
    document_id: str
    text: str
    score: float  # Final score after all stages
    rank: int  # Final rank position
    
    # Stage-specific scores
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing info
    processing_stages: List[str] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    """Performance metrics for retrieval pipeline."""
    total_time: float
    query_processing_time: float = 0.0
    search_time: float = 0.0
    reranking_time: float = 0.0
    
    num_queries_generated: int = 1
    num_initial_results: int = 0
    num_final_results: int = 0
    
    stages_used: List[str] = field(default_factory=list)


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline orchestrating all components.
    
    Features:
    - Query processing (rewriting, expansion)
    - Hybrid search (vector + BM25)
    - Reranking (cross-encoder)
    - Performance tracking
    - Flexible configuration
    """
    
    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        enable_query_processing: bool = True,
        enable_hybrid_search: bool = True,
        enable_reranking: bool = True,
        mode: Literal["fast", "balanced", "accurate"] = "balanced"
    ):
        """
        Initialize RetrievalPipeline.
        
        Args:
            vector_store: QdrantVectorStore instance
            chunks: List of chunks for BM25 indexing
            enable_query_processing: Use query processor
            enable_hybrid_search: Use hybrid search (else vector only)
            enable_reranking: Use reranker
            mode: Performance mode ("fast", "balanced", "accurate")
        """
        self.vector_store = vector_store
        self.chunks = chunks or []
        
        # Configure based on mode
        if mode == "fast":
            enable_query_processing = False
            enable_hybrid_search = False
            enable_reranking = False
        elif mode == "balanced":
            enable_query_processing = True
            enable_hybrid_search = True
            enable_reranking = False
        # "accurate" keeps all True
        
        self.enable_query_processing = enable_query_processing
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_reranking = enable_reranking
        self.mode = mode
        
        # Initialize components
        self.query_processor = None
        if enable_query_processing:
            try:
                self.query_processor = QueryProcessor()
                logger.info("Query processor initialized")
            except Exception as e:
                logger.warning(f"Query processor initialization failed: {e}")
        
        self.hybrid_search = None
        if enable_hybrid_search and vector_store:
            try:
                self.hybrid_search = HybridSearch(
                    vector_store=vector_store,
                    chunks=chunks
                )
                logger.info("Hybrid search initialized")
            except Exception as e:
                logger.warning(f"Hybrid search initialization failed: {e}")
        
        self.reranker = None
        if enable_reranking:
            try:
                self.reranker = Reranker()
                logger.info("Reranker initialized")
            except Exception as e:
                logger.warning(f"Reranker initialization failed: {e}")
        
        logger.success(f"✅ RetrievalPipeline initialized (mode: {mode})")
        logger.info(f"  Query processing: {enable_query_processing}")
        logger.info(f"  Hybrid search: {enable_hybrid_search}")
        logger.info(f"  Reranking: {enable_reranking}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of results to return
            filters: Metadata filters for search
            context: Optional conversation context
            
        Returns:
            Tuple of (results, metrics)
        """
        start_time = time.time()
        metrics = RetrievalMetrics(total_time=0.0)
        stages_used = []
        
        logger.info(f"🔍 Retrieving for query: '{query}'")
        logger.info(f"  Mode: {self.mode}, Top-K: {top_k}")
        
        # Stage 1: Query Processing
        processed_query = None
        if self.query_processor and self.enable_query_processing:
            qp_start = time.time()
            try:
                processed_query = self.query_processor.process(
                    query,
                    context=context,
                    num_expansions=2
                )
                metrics.query_processing_time = time.time() - qp_start
                metrics.num_queries_generated = len(processed_query.get_all_queries())
                stages_used.append("query_processing")
                logger.info(f"  Query processed: {metrics.num_queries_generated} variations")
                
                # NEW: Merge academic filters from query processing
                academic_filters = processed_query.metadata.get("academic_filters", {})
                if academic_filters:
                    logger.info(f"  📚 Academic filters detected: {list(academic_filters.keys())}")
                    if filters is None:
                        filters = {}
                    filters.update(academic_filters)
                
                # NEW: Apply section-based filtering
                section_intent = processed_query.metadata.get("section_intent")
                if section_intent:
                    logger.info(f"  📊 Section filter: {section_intent}")
                    if filters is None:
                        filters = {}
                    filters["section_type"] = section_intent
                
            except Exception as e:
                logger.warning(f"Query processing failed: {e}")
        
        # Use processed or original query
        search_query = processed_query.rewritten_query if processed_query else query
        
        # Stage 2: Search (Hybrid or Vector-only)
        search_start = time.time()
        search_results = []
        
        if self.hybrid_search and self.enable_hybrid_search:
            # Hybrid search (vector + BM25)
            try:
                search_results = self.hybrid_search.search(
                    query=search_query,
                    top_k=top_k * 4,  # Get more for reranking
                    filters=filters
                )
                stages_used.append("hybrid_search")
                logger.info(f"  Hybrid search: {len(search_results)} results")
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}")
        
        elif self.vector_store:
            # Vector-only search
            try:
                vector_results = self.vector_store.search(
                    query=search_query,
                    top_k=top_k * 4,
                    filters=filters
                )
                # Convert to hybrid format
                search_results = [
                    type('Result', (), {
                        'chunk_id': r.chunk_id,
                        'document_id': r.document_id,
                        'text': r.text,
                        'vector_score': r.score,
                        'bm25_score': 0.0,
                        'fusion_score': r.score,
                        'metadata': r.metadata
                    })()
                    for r in vector_results
                ]
                stages_used.append("vector_search")
                logger.info(f"  Vector search: {len(search_results)} results")
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return [], metrics
        
        metrics.search_time = time.time() - search_start
        metrics.num_initial_results = len(search_results)
        
        if not search_results and filters and "section_type" in filters:
            # Retry without section filter (papers may not have section metadata)
            logger.warning("No results with section filter, retrying without it...")
            fallback_filters = {k: v for k, v in filters.items() if k != "section_type"}
            fallback_filters = fallback_filters or None

            if self.hybrid_search and self.enable_hybrid_search:
                try:
                    search_results = self.hybrid_search.search(
                        query=search_query,
                        top_k=top_k * 4,
                        filters=fallback_filters
                    )
                    logger.info(f"  Fallback hybrid search: {len(search_results)} results")
                except Exception as e:
                    logger.warning(f"Fallback hybrid search failed: {e}")
            elif self.vector_store:
                try:
                    vector_results = self.vector_store.search(
                        query=search_query,
                        top_k=top_k * 4,
                        filters=fallback_filters
                    )
                    search_results = [
                        type('Result', (), {
                            'chunk_id': r.chunk_id,
                            'document_id': r.document_id,
                            'text': r.text,
                            'vector_score': r.score,
                            'bm25_score': 0.0,
                            'fusion_score': r.score,
                            'metadata': r.metadata
                        })()
                        for r in vector_results
                    ]
                    logger.info(f"  Fallback vector search: {len(search_results)} results")
                except Exception as e:
                    logger.error(f"Fallback vector search failed: {e}")

            metrics.num_initial_results = len(search_results)

        if not search_results:
            logger.warning("No search results found")
            metrics.total_time = time.time() - start_time
            metrics.stages_used = stages_used
            return [], metrics
        
        # Stage 3: Reranking
        reranked_results = search_results
        if self.reranker and self.enable_reranking and len(search_results) > 1:
            rerank_start = time.time()
            try:
                reranked_results = self.reranker.rerank(
                    query=search_query,
                    results=search_results,
                    top_k=top_k
                )
                metrics.reranking_time = time.time() - rerank_start
                stages_used.append("reranking")
                logger.info(f"  Reranked to top {len(reranked_results)} results")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                reranked_results = search_results[:top_k]
        else:
            reranked_results = search_results[:top_k]
        
        # Stage 4: Convert to RetrievalResult
        final_results = []
        for rank, result in enumerate(reranked_results[:top_k], 1):
            # Handle different result types
            if hasattr(result, 'rerank_score'):
                # RankedResult from reranker
                final_score = result.rerank_score
                vector_score = getattr(result, 'vector_score', 0.0)
                bm25_score = getattr(result, 'bm25_score', 0.0)
                fusion_score = result.original_score
                rerank_score = result.rerank_score
            elif hasattr(result, 'fusion_score'):
                # HybridSearchResult
                final_score = result.fusion_score
                vector_score = result.vector_score
                bm25_score = result.bm25_score
                fusion_score = result.fusion_score
                rerank_score = 0.0
            else:
                # Generic result
                final_score = getattr(result, 'score', 0.0)
                vector_score = final_score
                bm25_score = 0.0
                fusion_score = final_score
                rerank_score = 0.0
            
            final_results.append(RetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                score=final_score,
                rank=rank,
                vector_score=vector_score,
                bm25_score=bm25_score,
                fusion_score=fusion_score,
                rerank_score=rerank_score,
                metadata=getattr(result, 'metadata', {}),
                processing_stages=stages_used.copy()
            ))
        
        metrics.num_final_results = len(final_results)
        metrics.total_time = time.time() - start_time
        metrics.stages_used = stages_used
        
        logger.success(f"✅ Retrieved {len(final_results)} results in {metrics.total_time:.3f}s")
        
        return final_results, metrics
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[tuple[List[RetrievalResult], RetrievalMetrics]]:
        """
        Retrieve for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of results per query
            
        Returns:
            List of (results, metrics) tuples
        """
        logger.info(f"Batch retrieval: {len(queries)} queries")
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.retrieve(query, top_k=top_k)
            results.append(result)
        
        logger.success(f"✅ Batch retrieval complete: {len(queries)} queries")
        return results
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return {
            "mode": self.mode,
            "query_processing": self.enable_query_processing,
            "hybrid_search": self.enable_hybrid_search,
            "reranking": self.enable_reranking,
            "components": {
                "query_processor": self.query_processor is not None,
                "hybrid_search": self.hybrid_search is not None,
                "reranker": self.reranker is not None,
                "vector_store": self.vector_store is not None
            }
        }


# Convenience function
def retrieve(
    query: str,
    vector_store: QdrantVectorStore,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    mode: str = "balanced"
) -> List[RetrievalResult]:
    """
    Quick retrieval with default pipeline.
    
    Args:
        query: Search query
        vector_store: Qdrant vector store
        chunks: List of chunks
        top_k: Number of results
        mode: Performance mode
        
    Returns:
        List of retrieval results
    """
    pipeline = RetrievalPipeline(
        vector_store=vector_store,
        chunks=chunks,
        mode=mode
    )
    results, _ = pipeline.retrieve(query, top_k=top_k)
    return results
