"""
Hybrid Search - Combine vector search (semantic) with BM25 (keyword).

This is Step 2 of Phase 3: Query Processing & Retrieval.

PURPOSE:
--------
Combine two complementary search methods:
1. **Vector Search**: Find semantically similar chunks (meaning-based)
2. **BM25 Search**: Find keyword matches (term-based)
3. **Fusion**: Merge results using Reciprocal Rank Fusion (RRF)

WHY HYBRID SEARCH?
------------------
Vector search alone has limitations:
- May miss exact keyword matches
- Can drift from specific terms
- Might return semantically similar but irrelevant results

BM25 search alone has limitations:
- Misses semantic similarity
- Requires exact term matches
- No understanding of meaning

Hybrid = Best of both worlds! 🚀

RECIPROCAL RANK FUSION (RRF):
------------------------------
Merges ranked lists from multiple search methods.

Formula: score(chunk) = Σ 1 / (k + rank_i)
- k = 60 (standard constant)
- rank_i = rank in each search result list

Example:
- Vector search: [A, B, C, D]  (A=rank1, B=rank2, ...)
- BM25 search:    [B, D, A, E]  (B=rank1, D=rank2, ...)
  
RRF scores:
- A: 1/61 + 1/63 = 0.032
- B: 1/62 + 1/61 = 0.032  
- C: 1/63 + 0 = 0.016
- D: 1/64 + 1/62 = 0.032
- E: 0 + 1/64 = 0.016

Sorted by RRF score: [A, B, D, C, E]

PERFORMANCE IMPROVEMENTS:
-------------------------
Hybrid search typically gives:
- 15-30% better recall
- 10-20% better precision
- More robust to query variations
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re

from rank_bm25 import BM25Okapi
from loguru import logger

from docusense.vectorstore import QdrantVectorStore, SearchResult


@dataclass
class HybridSearchResult:
    """Result from hybrid search with fusion scores."""
    chunk_id: str
    document_id: str
    text: str
    vector_score: float = 0.0  # Cosine similarity from vector search
    bm25_score: float = 0.0    # BM25 score from keyword search
    fusion_score: float = 0.0  # Combined RRF score
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def final_score(self) -> float:
        """Return the fusion score as final score."""
        return self.fusion_score


class HybridSearch:
    """
    Hybrid search combining vector and BM25 keyword search.
    
    Features:
    - Vector search via Qdrant (semantic similarity)
    - BM25 search on chunk corpus (keyword matching)
    - Reciprocal Rank Fusion (RRF) for result merging
    - Configurable weights for each method
    """
    
    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60
    ):
        """
        Initialize HybridSearch.
        
        Args:
            vector_store: QdrantVectorStore for vector search
            chunks: List of chunk dicts for BM25 indexing
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            rrf_k: RRF constant (default 60)
        """
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        
        # Initialize BM25 index
        self.bm25_index = None
        self.chunks = []
        self.chunk_map = {}  # chunk_id -> chunk dict
        
        if chunks:
            self.index_chunks(chunks)
        
        logger.info("HybridSearch initialized")
        logger.info(f"  Vector weight: {vector_weight}")
        logger.info(f"  BM25 weight: {bm25_weight}")
        logger.info(f"  RRF k: {rrf_k}")
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Index chunks for BM25 search.
        
        Args:
            chunks: List of chunk dicts with 'text', 'chunk_id', etc.
        """
        logger.info(f"Indexing {len(chunks)} chunks for BM25...")
        
        self.chunks = chunks
        
        # Build chunk map for quick lookup
        self.chunk_map = {chunk['chunk_id']: chunk for chunk in chunks}
        
        # Tokenize texts for BM25
        tokenized_texts = [self._tokenize(chunk['text']) for chunk in chunks]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_texts)
        
        logger.success(f"✅ BM25 index built with {len(chunks)} chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens (lowercased words)
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        use_vector: bool = True,
        use_bm25: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector and BM25.
        
        Args:
            query: Search query
            top_k: Final number of results after fusion
            vector_top_k: Number of results from vector search (default: top_k * 2)
            bm25_top_k: Number of results from BM25 search (default: top_k * 2)
            use_vector: Enable vector search
            use_bm25: Enable BM25 search
            filters: Metadata filters for vector search
            
        Returns:
            List of HybridSearchResult sorted by fusion score
        """
        logger.info(f"Hybrid search: '{query}'")
        logger.info(f"  Vector: {use_vector}, BM25: {use_bm25}, Top-K: {top_k}")
        
        # Default: retrieve more candidates for better fusion
        vector_top_k = vector_top_k or top_k * 2
        bm25_top_k = bm25_top_k or top_k * 2
        
        vector_results = []
        bm25_results = []
        
        # 1. Vector Search
        if use_vector and self.vector_store:
            try:
                vector_results = self.vector_store.search(
                    query=query,
                    top_k=vector_top_k,
                    filters=filters
                )
                logger.info(f"  Vector search: {len(vector_results)} results")
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # 2. BM25 Search
        if use_bm25 and self.bm25_index:
            try:
                bm25_results = self._bm25_search(query, bm25_top_k)
                logger.info(f"  BM25 search: {len(bm25_results)} results")
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")
        
        # 3. Merge with Reciprocal Rank Fusion
        if vector_results and bm25_results:
            merged = self._reciprocal_rank_fusion(vector_results, bm25_results, top_k)
            logger.info(f"  RRF fusion: {len(merged)} results")
        elif vector_results:
            merged = self._convert_vector_results(vector_results[:top_k])
            logger.info(f"  Vector only: {len(merged)} results")
        elif bm25_results:
            merged = self._convert_bm25_results(bm25_results[:top_k])
            logger.info(f"  BM25 only: {len(merged)} results")
        else:
            logger.warning("No results from either search method")
            merged = []
        
        logger.success(f"✅ Hybrid search complete: {len(merged)} results")
        return merged
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk, bm25_score) tuples
        """
        if not self.bm25_index:
            logger.warning("BM25 index not initialized")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = [(self.chunks[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[Tuple[Dict[str, Any], float]],
        top_k: int
    ) -> List[HybridSearchResult]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of final results
            
        Returns:
            List of HybridSearchResult sorted by RRF score
        """
        # Track all chunks and their scores
        chunk_scores = defaultdict(lambda: {
            'chunk_id': None,
            'document_id': None,
            'text': None,
            'vector_score': 0.0,
            'bm25_score': 0.0,
            'vector_rank': None,
            'bm25_rank': None,
            'metadata': {}
        })
        
        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk_id
            chunk_scores[chunk_id].update({
                'chunk_id': chunk_id,
                'document_id': result.document_id,
                'text': result.text,
                'vector_score': result.score,
                'vector_rank': rank,
                'metadata': result.metadata
            })
        
        # Process BM25 results
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            chunk_id = chunk.get('chunk_id') or chunk.get('id')
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]['bm25_score'] = score
                chunk_scores[chunk_id]['bm25_rank'] = rank
            else:
                chunk_scores[chunk_id].update({
                    'chunk_id': chunk_id,
                    'document_id': chunk.get('document_id', ''),
                    'text': chunk.get('text', ''),
                    'bm25_score': score,
                    'bm25_rank': rank,
                    'metadata': chunk.get('metadata', {})
                })
        
        # Calculate RRF scores
        for chunk_id, data in chunk_scores.items():
            rrf_score = 0.0
            
            # Add vector contribution
            if data['vector_rank']:
                rrf_score += self.vector_weight / (self.rrf_k + data['vector_rank'])
            
            # Add BM25 contribution
            if data['bm25_rank']:
                rrf_score += self.bm25_weight / (self.rrf_k + data['bm25_rank'])
            
            data['fusion_score'] = rrf_score
        
        # Convert to HybridSearchResult and sort by fusion score
        results = [
            HybridSearchResult(
                chunk_id=data['chunk_id'],
                document_id=data['document_id'],
                text=data['text'],
                vector_score=data['vector_score'],
                bm25_score=data['bm25_score'],
                fusion_score=data['fusion_score'],
                metadata=data['metadata']
            )
            for data in chunk_scores.values()
        ]
        
        # Sort by fusion score (descending)
        results.sort(key=lambda r: r.fusion_score, reverse=True)
        
        return results[:top_k]
    
    def _convert_vector_results(self, results: List[SearchResult]) -> List[HybridSearchResult]:
        """Convert vector-only results to HybridSearchResult."""
        return [
            HybridSearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text,
                vector_score=r.score,
                fusion_score=r.score,
                metadata=r.metadata
            )
            for r in results
        ]
    
    def _convert_bm25_results(self, results: List[Tuple[Dict[str, Any], float]]) -> List[HybridSearchResult]:
        """Convert BM25-only results to HybridSearchResult."""
        return [
            HybridSearchResult(
                chunk_id=chunk.get('chunk_id') or chunk.get('id'),
                document_id=chunk.get('document_id', ''),
                text=chunk.get('text', ''),
                bm25_score=score,
                fusion_score=score,
                metadata=chunk.get('metadata', {})
            )
            for chunk, score in results
        ]


# Convenience function
def hybrid_search(
    query: str,
    vector_store: QdrantVectorStore,
    chunks: List[Dict[str, Any]],
    top_k: int = 5
) -> List[HybridSearchResult]:
    """
    Quick hybrid search with default settings.
    
    Args:
        query: Search query
        vector_store: Qdrant vector store
        chunks: List of chunks for BM25
        top_k: Number of results
        
    Returns:
        List of hybrid search results
    """
    searcher = HybridSearch(vector_store=vector_store, chunks=chunks)
    return searcher.search(query, top_k=top_k)
