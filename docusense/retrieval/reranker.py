"""
Reranker - Re-score search results using cross-encoder for better precision.

This is Step 3 of Phase 3: Query Processing & Retrieval.

PURPOSE:
--------
Improve search precision by re-scoring initial results with a more accurate model:
1. **First-stage retrieval**: Fast bi-encoder (vector search) gets candidates
2. **Second-stage reranking**: Slow but accurate cross-encoder re-scores top results
3. **Final ranking**: Return best results after reranking

WHY RERANKING?
--------------
Two-stage retrieval is industry standard:

**Stage 1 - Bi-Encoder (Current):**
- Encodes query and documents separately
- Fast: Can search millions of docs
- Less accurate: No query-document interaction
- Used in: Vector search, initial retrieval

**Stage 2 - Cross-Encoder (Reranker):**
- Encodes query+document together
- Slow: Can only score hundreds
- More accurate: Captures query-doc interaction
- Used in: Final reranking of top candidates

PERFORMANCE IMPROVEMENT:
------------------------
Reranking typically improves:
- Precision@5: +20-40%
- nDCG: +10-25%
- User satisfaction: +15-30%

EXAMPLE:
--------
Query: "How does machine learning work?"

Initial retrieval (100 candidates):
1. "ML is a subset of AI..." (score: 0.85)
2. "Neural networks learn patterns..." (score: 0.83)
3. "The ML process involves..." (score: 0.81)
...

After reranking (top 5):
1. "The ML process involves..." (NEW score: 0.92) ⬆️
2. "ML is a subset of AI..." (NEW score: 0.89) ⬇️
3. "Neural networks learn patterns..." (NEW score: 0.87) ⬇️

Notice how cross-encoder reordered results for better relevance!

FREE MODEL:
-----------
Uses: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on Microsoft MARCO dataset
- 22M parameters (small & fast)
- No API costs
- Runs locally
"""

from typing import List, Union, Optional
from dataclasses import dataclass
import time

from loguru import logger

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Reranking disabled.")

from docusense.config.settings import settings


@dataclass
class RankedResult:
    """Result after reranking with updated score."""
    chunk_id: str
    document_id: str
    text: str
    original_score: float  # Score from initial retrieval
    rerank_score: float    # Score from cross-encoder
    rank_change: int = 0   # Position change after reranking
    metadata: dict = None

    # Carried through from the search stage. Reranking replaces the ordering,
    # not the record of how a chunk was found, and callers report the two
    # signals separately -- without these the per-stage scores read 0.0 for
    # every result as soon as reranking is enabled.
    vector_score: float = 0.0
    bm25_score: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Reranker:
    """
    Rerank search results using cross-encoder for better precision.
    
    Features:
    - Cross-encoder scoring (query + document interaction)
    - Configurable reranking depth
    - Performance tracking (time, score changes)
    - Fallback mode if cross-encoder unavailable
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 512
    ):
        """
        Initialize Reranker.
        
        Args:
            model_name: Cross-encoder model name (default from settings)
            device: Device to use ('cpu' or 'cuda')
            max_length: Maximum sequence length
        """
        self.model_name = model_name or settings.reranker_model
        self.device = device
        self.max_length = max_length
        self.model = None
        
        # Load cross-encoder model
        if CROSS_ENCODER_AVAILABLE:
            try:
                logger.info(f"Loading cross-encoder: {self.model_name}")
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=max_length,
                    device=device
                )
                logger.success(f"✅ Reranker loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                logger.warning("Reranking will be disabled")
        else:
            logger.warning("Reranker initialization skipped (sentence-transformers not available)")
    
    def rerank(
        self,
        query: str,
        results: List[Union[dict, RankedResult]],
        top_k: Optional[int] = None
    ) -> List[RankedResult]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Search query
            results: List of search results (dicts or RankedResult objects)
            top_k: Number of results to return after reranking (None = return all)
            
        Returns:
            List of RankedResult sorted by rerank score
        """
        if not results:
            logger.warning("No results to rerank")
            return []
        
        logger.info(f"Reranking {len(results)} results for query: '{query}'")
        start_time = time.time()
        
        # Convert results to standardized format
        standardized_results = self._standardize_results(results)
        
        # If cross-encoder not available, return original order
        if not self.model:
            logger.warning("Cross-encoder not available, returning original order")
            return standardized_results[:top_k] if top_k else standardized_results
        
        # Prepare query-document pairs
        pairs = [(query, result.text) for result in standardized_results]
        
        # Get cross-encoder scores
        try:
            scores = self.model.predict(pairs)
            logger.info(f"Cross-encoder scored {len(pairs)} pairs")
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return standardized_results[:top_k] if top_k else standardized_results
        
        # Update results with rerank scores
        for result, score in zip(standardized_results, scores):
            result.rerank_score = float(score)
        
        # Sort by rerank score (descending)
        original_order = {r.chunk_id: i for i, r in enumerate(standardized_results)}
        reranked = sorted(standardized_results, key=lambda r: r.rerank_score, reverse=True)
        
        # Calculate rank changes
        for new_rank, result in enumerate(reranked):
            old_rank = original_order[result.chunk_id]
            result.rank_change = old_rank - new_rank  # Positive = moved up
        
        # Log statistics
        elapsed = time.time() - start_time
        logger.info(f"Reranking complete in {elapsed:.3f}s")
        
        # Log significant rank changes
        big_changes = [r for r in reranked if abs(r.rank_change) >= 3]
        if big_changes:
            logger.info(f"Significant rank changes: {len(big_changes)} results moved 3+ positions")
        
        # Return top-k or all
        result_count = top_k if top_k else len(reranked)
        logger.success(f"✅ Returned top {result_count} reranked results")
        
        return reranked[:top_k] if top_k else reranked
    
    def _standardize_results(self, results: List[Union[dict, RankedResult]]) -> List[RankedResult]:
        """
        Convert various result formats to RankedResult.
        
        Args:
            results: List of results (dicts, SearchResult, HybridSearchResult, etc.)
            
        Returns:
            List of RankedResult objects
        """
        standardized = []
        
        for result in results:
            if isinstance(result, RankedResult):
                standardized.append(result)
            elif isinstance(result, dict):
                # Dictionary format
                standardized.append(RankedResult(
                    chunk_id=result.get('chunk_id', ''),
                    document_id=result.get('document_id', ''),
                    text=result.get('text', ''),
                    original_score=result.get('score', 0.0),
                    rerank_score=0.0,
                    metadata=result.get('metadata', {}),
                    vector_score=result.get('vector_score', 0.0),
                    bm25_score=result.get('bm25_score', 0.0)
                ))
            elif hasattr(result, 'chunk_id') and hasattr(result, 'text'):
                # Object with chunk_id and text attributes
                standardized.append(RankedResult(
                    chunk_id=result.chunk_id,
                    document_id=getattr(result, 'document_id', ''),
                    text=result.text,
                    original_score=getattr(result, 'score', getattr(result, 'fusion_score', 0.0)),
                    rerank_score=0.0,
                    metadata=getattr(result, 'metadata', {}),
                    vector_score=getattr(result, 'vector_score', 0.0),
                    bm25_score=getattr(result, 'bm25_score', 0.0)
                ))
            else:
                logger.warning(f"Unknown result format: {type(result)}, skipping")
        
        return standardized
    
    def get_score_distribution(self, results: List[RankedResult]) -> dict:
        """
        Analyze score distribution of reranked results.
        
        Args:
            results: List of reranked results
            
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {}
        
        rerank_scores = [r.rerank_score for r in results]
        original_scores = [r.original_score for r in results]
        rank_changes = [r.rank_change for r in results]
        
        return {
            'count': len(results),
            'rerank': {
                'min': min(rerank_scores),
                'max': max(rerank_scores),
                'mean': sum(rerank_scores) / len(rerank_scores),
            },
            'original': {
                'min': min(original_scores),
                'max': max(original_scores),
                'mean': sum(original_scores) / len(original_scores),
            },
            'rank_changes': {
                'max_up': max(rank_changes),
                'max_down': min(rank_changes),
                'mean_abs': sum(abs(c) for c in rank_changes) / len(rank_changes),
            }
        }


# Convenience function
def rerank(
    query: str,
    results: List[Union[dict, RankedResult]],
    top_k: int = 5
) -> List[RankedResult]:
    """
    Quick reranking with default settings.
    
    Args:
        query: Search query
        results: List of search results
        top_k: Number of results to return
        
    Returns:
        List of reranked results
    """
    reranker = Reranker()
    return reranker.rerank(query, results, top_k=top_k)
