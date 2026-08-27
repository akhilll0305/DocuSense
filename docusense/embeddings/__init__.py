"""
Embeddings module for DocuSense.

Exports:
- EmbeddingGenerator: Generate embeddings from text
- EmbeddingResult: Result dataclass
- embed_text: Convenience function for single text
- embed_batch: Convenience function for batch
"""

from .embedding_generator import (
    EmbeddingGenerator,
    EmbeddingResult,
    embed_text,
    embed_batch
)

__all__ = [
    'EmbeddingGenerator',
    'EmbeddingResult',
    'embed_batch',
    'embed_text'
]
