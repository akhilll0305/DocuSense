"""
Embedding Generator - Creates vector embeddings from text using sentence-transformers.

This is Step 1 of Phase 2: Vector Embeddings & Search.

PURPOSE:
--------
Convert text chunks into dense vector embeddings for semantic search:
1. Load embedding model (sentence-transformers - FREE, local)
2. Generate embeddings for text chunks
3. Batch processing for efficiency
4. Normalize vectors for cosine similarity

WHY EMBEDDINGS?
---------------
Embeddings capture SEMANTIC MEANING, not just keywords:

Keyword search (BM25):
  Query: "machine learning accuracy"
  Matches: Documents with exact words "machine", "learning", "accuracy"
  Misses: Documents about "model performance", "prediction quality"

Semantic search (embeddings):
  Query: "machine learning accuracy"
  Matches: Documents about ML performance, model quality, prediction metrics
  → Understanding meaning, not just matching words!

EMBEDDING MODEL CHOICE:
-----------------------
We use sentence-transformers (Hugging Face) because:
- FREE and open source
- Runs locally (no API calls, no costs)
- Fast inference (even on CPU)
- Pre-trained on semantic similarity tasks
- Multiple model options

Model Options (all FREE):
1. all-MiniLM-L6-v2 (384 dim) - RECOMMENDED
   - Fast: ~14ms per sentence on CPU
   - Good quality: 68.06 average performance
   - Small: 80 MB model size
   - Best for: General purpose, speed matters

2. all-mpnet-base-v2 (768 dim) - BEST QUALITY
   - Slower: ~45ms per sentence on CPU
   - Best quality: 69.57 average performance
   - Larger: 420 MB model size
   - Best for: When quality > speed

3. multi-qa-MiniLM-L6-cos-v1 (384 dim) - OPTIMIZED FOR Q&A
   - Trained specifically for question-answer pairs
   - Good for RAG systems
   - Similar speed to MiniLM-L6-v2

TOKEN LIMITS:
-------------
Most sentence-transformers models have a 512 token limit.
Text longer than 512 tokens will be truncated!
→ This is why we chunk documents BEFORE embedding!

EMBEDDING DIMENSION:
--------------------
- 384 dimensions (MiniLM): Good balance of speed/quality
- 768 dimensions (MPNet): Better quality, slower, more storage
- More dimensions ≠ always better (diminishing returns)

NORMALIZATION:
--------------
We normalize embeddings to unit length (L2 norm = 1) for:
- Cosine similarity (faster: dot product instead of cosine)
- Consistent distance metrics
- Better clustering
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from docusense.config.settings import settings


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: np.ndarray  # Shape: (num_texts, embedding_dim)
    model_name: str
    dimension: int
    num_texts: int
    

class EmbeddingGenerator:
    """
    Generate embeddings using sentence-transformers.
    
    Features:
    - FREE local embedding models
    - Batch processing for efficiency
    - GPU support (if available)
    - Automatic normalization
    - Progress tracking
    
    Usage:
        generator = EmbeddingGenerator()
        
        # Single text
        embedding = generator.embed_text("This is a test sentence")
        
        # Batch of texts
        embeddings = generator.embed_batch([
            "First sentence",
            "Second sentence",
            "Third sentence"
        ])
        
        # With progress bar
        embeddings = generator.embed_batch(
            texts,
            show_progress=True,
            description="Embedding chunks"
        )
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence-transformers model (default from settings)
            device: Device to use ["cpu", "cuda", "mps"] (default from settings)
            normalize: Normalize embeddings to unit length (default True)
            batch_size: Batch size for processing (default from settings)
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.normalize = normalize if normalize is not None else settings.embedding_normalize
        self.batch_size = batch_size or settings.embedding_batch_size
        
        logger.info("Initializing EmbeddingGenerator...")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Normalize: {self.normalize}")
        
        # Load model
        try:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.success(f"✅ Model loaded: {self.embedding_dim} dimensions")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
        
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
        description: str = "Embedding texts"
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            show_progress: Show progress bar (default False)
            description: Progress bar description
        
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return np.zeros((0, self.embedding_dim))
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        if not valid_texts:
            return np.zeros((0, self.embedding_dim))
        
        logger.info(f"Generating embeddings for {len(valid_texts)} texts...")
        
        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.success(f"✅ Generated {len(embeddings)} embeddings")
        
        return embeddings
    
    def embed_chunks_from_db(
        self,
        chunks: List[dict],
        text_field: str = "text"
    ) -> EmbeddingResult:
        """
        Generate embeddings for chunks retrieved from database.
        
        Args:
            chunks: List of chunk dicts (each has 'text' field)
            text_field: Name of text field in chunk dict
        
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        texts = [chunk.get(text_field, "") for chunk in chunks]
        
        embeddings = self.embed_batch(
            texts,
            show_progress=True,
            description="Embedding chunks"
        )
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            dimension=self.embedding_dim,
            num_texts=len(embeddings)
        )
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict with model metadata
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length,
            'normalize': self.normalize,
            'batch_size': self.batch_size
        }
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query vector (embedding_dim,)
            document_embeddings: Document vectors (num_docs, embedding_dim)
        
        Returns:
            Similarity scores (num_docs,) in range [0, 1]
        """
        # Ensure normalized (for dot product = cosine similarity)
        if not self.normalize:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            document_embeddings = document_embeddings / np.linalg.norm(
                document_embeddings, axis=1, keepdims=True
            )
        
        # Dot product (since normalized, this IS cosine similarity)
        similarities = np.dot(document_embeddings, query_embedding)
        
        # Clip to [0, 1] (handle floating point errors)
        similarities = np.clip(similarities, 0.0, 1.0)
        
        return similarities


# Convenience function for quick embedding
def embed_text(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Quick convenience function to embed a single text.
    
    Args:
        text: Text to embed
        model_name: Optional model name (default from settings)
    
    Returns:
        Embedding vector
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.embed_text(text)


def embed_batch(texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
    """
    Quick convenience function to embed multiple texts.
    
    Args:
        texts: List of texts to embed
        model_name: Optional model name (default from settings)
    
    Returns:
        Embedding matrix
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.embed_batch(texts, show_progress=True)
