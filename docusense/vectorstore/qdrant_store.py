"""
Qdrant Vector Store - Semantic search using Qdrant vector database.

This is Step 2 of Phase 2: Vector Embeddings & Search.

PURPOSE:
--------
Store and search document chunk embeddings using Qdrant:
1. Create collection with proper configuration
2. Store embeddings with metadata
3. Semantic search (find similar chunks)
4. Metadata filtering (by document, date, etc.)
5. Hybrid search (vector + keywords)

WHY QDRANT?
-----------
Better than FAISS for production:
- ✅ Persistent storage (survives restarts)
- ✅ Metadata filtering (filter by document_id, has_code, etc.)
- ✅ REST API (easy integration)
- ✅ Scalable (billions of vectors)
- ✅ Free and open source
- ✅ Easy setup (Docker or in-memory)

Qdrant Modes:
1. Memory mode: Fast, no persistence (testing)
2. Disk mode: Persistent, local storage (development)
3. Server mode: Remote Qdrant server (production)

VECTOR SEARCH STRATEGIES:
--------------------------
1. **Pure semantic search**: Find chunks with similar meaning
   - Query: "How does the algorithm work?"
   - Matches chunks about algorithm implementation
   
2. **Metadata filtering**: Semantic search WITHIN constraints
   - Query: "performance metrics" + filter: document_id="report_2024"
   - Only searches within that specific document
   
3. **Hybrid search**: Combine vector search + keyword search
   - Vector: Semantic meaning
   - Keyword (BM25): Exact term matches
   - Fusion: Best of both worlds

DISTANCE METRICS:
-----------------
- **Cosine** (RECOMMENDED): Measures angle between vectors
  - Range: 0 (identical) to 2 (opposite)
  - Best for: Semantic similarity
  - Required: Normalized embeddings

- **Dot product**: Inner product of vectors
  - Faster than cosine
  - Requires normalized vectors

- **Euclidean**: L2 distance
  - Not recommended for semantic search
  - Sensitive to magnitude
"""

from typing import List, Dict, Optional, Union, Literal, Any
from dataclasses import dataclass
from pathlib import Path
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    ScoredPoint,
    PayloadSchemaType,
    TextIndexParams
)
from loguru import logger

from docusense.config.settings import settings
from docusense.embeddings import EmbeddingGenerator


@dataclass
class SearchResult:
    """Result of a vector search query."""
    chunk_id: str
    document_id: str
    text: str
    score: float  # Similarity score (0-1, higher = more similar)
    metadata: Dict[str, Any]
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return f"[Score: {self.score:.3f}] {self.text[:100]}..."


class QdrantVectorStore:
    """
    Qdrant-based vector store for semantic search.
    
    Features:
    - Multiple modes (memory/disk/server)
    - Automatic collection creation
    - Batch upsert for efficiency
    - Metadata filtering
    - Score thresholding
    
    Usage:
        # Initialize
        vector_store = QdrantVectorStore()
        vector_store.create_collection()
        
        # Add chunks
        vector_store.add_chunks(chunks_with_embeddings)
        
        # Search
        results = vector_store.search(
            query="machine learning accuracy",
            top_k=5,
            filters={"document_id": "doc_123"}
        )
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        mode: Optional[Literal["memory", "disk", "server"]] = None,
        path: Optional[Path] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            collection_name: Name of collection (default from settings)
            mode: Qdrant mode ["memory", "disk", "server"]
            path: Path for disk mode
            url: URL for server mode
            api_key: API key for cloud Qdrant
            embedding_generator: EmbeddingGenerator instance (creates new if None)
        """
        self.collection_name = collection_name or settings.qdrant_collection_name
        # Auto-detect server mode if credentials provided
        self.mode = mode or settings.effective_qdrant_mode
        self.path = path or settings.qdrant_path
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        
        # Initialize embedding generator
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.embedding_dim = self.embedding_generator.embedding_dim
        
        logger.info(f"Initializing QdrantVectorStore...")
        logger.info(f"  Collection: {self.collection_name}")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        
        # Initialize Qdrant client based on mode
        try:
            if self.mode == "memory":
                logger.info("  Storage: In-memory (no persistence)")
                self.client = QdrantClient(":memory:")
            
            elif self.mode == "disk":
                self.path.mkdir(parents=True, exist_ok=True)
                logger.info(f"  Storage: Disk at {self.path}")
                self.client = QdrantClient(path=str(self.path))
            
            elif self.mode == "server":
                if not self.url:
                    raise ValueError("Server mode requires 'url' parameter")
                logger.info(f"  Storage: Remote server at {self.url}")
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key
                )
            
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            
            logger.success(f"✅ Connected to Qdrant ({self.mode} mode)")
        
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise
    
    def _get_distance_metric(self, distance: Optional[Union[Distance, str]] = None) -> Distance:
        """
        Convert distance metric string to Distance enum.
        
        Args:
            distance: Distance metric (Distance enum or string)
            
        Returns:
            Distance enum
        """
        if isinstance(distance, Distance):
            return distance
        
        # Use provided string or fallback to settings
        distance_str = distance or settings.distance_metric
        
        # Convert string to Distance enum
        distance_map = {
            "COSINE": Distance.COSINE,
            "EUCLIDEAN": Distance.EUCLID,
            "DOT": Distance.DOT
        }
        
        if distance_str not in distance_map:
            logger.warning(f"Unknown distance metric: {distance_str}, using COSINE")
            return Distance.COSINE
        
        return distance_map[distance_str]
    
    def create_collection(
        self,
        recreate: bool = False,
        distance: Optional[Union[Distance, str]] = None
    ):
        """
        Create Qdrant collection with proper configuration.
        
        Args:
            recreate: Delete existing collection and recreate (default False)
            distance: Distance metric (Distance enum, string, or None for settings default)
        """
        # Get distance metric
        distance_metric = self._get_distance_metric(distance)
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists:
            if recreate:
                logger.warning(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
        
        # Create collection
        logger.info(f"Creating collection: {self.collection_name}")
        logger.info(f"  Embedding dimension: {self.embedding_dim}")
        logger.info(f"  Distance metric: {distance_metric}")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=distance_metric
            )
        )
        
        # Create payload indexes for filterable fields
        # This enables efficient filtering by document_id, chunk_id, etc.
        logger.info("Creating payload indexes...")
        
        # Index for document_id (keyword for exact matching)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        # Index for chunk_id (keyword for exact matching)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="chunk_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        # Index for has_code (bool for filtering code chunks)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="has_code",
            field_schema=PayloadSchemaType.BOOL
        )
        
        # Index for has_tables (bool for filtering table chunks)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="has_tables",
            field_schema=PayloadSchemaType.BOOL
        )
        
        # Index for text (full-text search)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="text",
            field_schema=PayloadSchemaType.TEXT
        )
        
        # ================================================================
        # RESEARCH PAPER INDEXES (NEW!)
        # ================================================================
        logger.info("Creating research paper payload indexes...")
        
        # Index for paper_title (keyword for exact/prefix matching)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="paper_title",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        # Index for authors (keyword array for filtering by author)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="authors",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        # Index for year (integer for range queries: 2020-2024)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="year",
            field_schema=PayloadSchemaType.INTEGER
        )
        
        # Index for section_type (keyword: abstract, methodology, results, etc.)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="section_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        # Index for venue (keyword: conference/journal name)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="venue",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        # Index for paper_type (keyword: conference, journal, arxiv)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="paper_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        # Index for has_equations (bool for filtering chunks with math)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="has_equations",
            field_schema=PayloadSchemaType.BOOL
        )
        
        # Index for has_citations (bool for filtering chunks with references)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="has_citations",
            field_schema=PayloadSchemaType.BOOL
        )

        logger.success(
            f"✅ Collection '{self.collection_name}' created with "
            f"standard + research paper payload indexes"
        )
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dicts with 'text', 'chunk_id', 'document_id', metadata
            batch_size: Batch size for upsert (default 100)
        
        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return 0
        
        logger.info(f"Adding {len(chunks)} chunks to Qdrant...")
        
        # Generate embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_generator.embed_batch(
            texts,
            show_progress=True,
            description="Generating embeddings"
        )
        
        # Prepare points for Qdrant
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate UUID for Qdrant point ID (required by Qdrant)
            # Store original chunk_id in payload for tracking
            point_id = str(uuid.uuid4())
            
            # Prepare payload (metadata)
            payload = {
                'chunk_id': chunk.get('chunk_id'),
                'document_id': chunk.get('document_id'),
                'text': chunk['text'],
                'chunk_index': chunk.get('chunk_index', idx),
                'header_path': chunk.get('header_path', ''),
                'token_count': chunk.get('token_count', 0),
                'has_code': chunk.get('has_code', False),
                'has_tables': chunk.get('has_tables', False),
            }
            
            # Add any additional metadata
            if 'metadata' in chunk:
                payload.update(chunk['metadata'])
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            ))
        
        # Upsert points in batches
        logger.info(f"Upserting {len(points)} points...")
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.success(f"✅ Added {len(points)} chunks to Qdrant")
        return len(points)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Semantic search for similar chunks.
        
        Args:
            query: Search query text
            top_k: Number of results to return (default 5)
            score_threshold: Minimum similarity score (0-1, optional)
            filters: Metadata filters (e.g., {"document_id": "doc_123"})
        
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        logger.info(f"Searching for: '{query}'")
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Prepare filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            qdrant_filter = Filter(must=conditions)
        
        # Apply score threshold from settings if not provided
        if score_threshold is None and settings.use_score_threshold:
            score_threshold = settings.similarity_threshold
        
        # Search using the new query_points API (Qdrant v1.7+)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold
        ).points
        
        # Convert to SearchResult objects
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result.payload.get('chunk_id', str(result.id)),
                document_id=result.payload.get('document_id', ''),
                text=result.payload.get('text', ''),
                score=result.score,
                metadata=result.payload
            ))
        
        logger.info(f"Found {len(search_results)} results")
        return search_results
    
    def search_by_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search within a specific document.
        
        Args:
            query: Search query
            document_id: Document ID to search within
            top_k: Number of results
        
        Returns:
            List of SearchResult objects
        """
        return self.search(
            query=query,
            top_k=top_k,
            filters={"document_id": document_id}
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dict with collection stats
        """
        info = self.client.get_collection(self.collection_name)
        
        return {
            'collection_name': self.collection_name,
            'points_count': info.points_count,
            'vectors_count': info.points_count,  # points_count is the number of vectors
            'segments_count': info.segments_count,
            'status': info.status.value if hasattr(info.status, 'value') else str(info.status),
            'embedding_dimension': self.embedding_dim
        }
    
    def delete_by_document(self, document_id: str) -> int:
        """
        Delete all chunks from a specific document.
        
        Args:
            document_id: Document ID to delete
        
        Returns:
            Number of points deleted (approximate)
        """
        logger.info(f"Deleting chunks for document: {document_id}")
        
        # Delete points matching document_id
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
        
        logger.success(f"✅ Deleted chunks for document: {document_id}")
        return 1  # Qdrant doesn't return count
    
    def delete_collection(self):
        """Delete the entire collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.info("Collection deleted")


# Convenience function for quick search
def search_documents(
    query: str,
    top_k: int = 5,
    collection_name: Optional[str] = None
) -> List[SearchResult]:
    """
    Quick convenience function for semantic search.
    
    Args:
        query: Search query
        top_k: Number of results
        collection_name: Optional collection name
    
    Returns:
        List of SearchResult objects
    """
    vector_store = QdrantVectorStore(collection_name=collection_name)
    return vector_store.search(query, top_k=top_k)
