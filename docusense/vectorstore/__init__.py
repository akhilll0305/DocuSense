"""
Vector store module for DocuSense.

Exports:
- QdrantVectorStore: Qdrant-based vector store
- SearchResult: Search result dataclass
- search_documents: Convenience function
"""

from .qdrant_store import (
    QdrantVectorStore,
    SearchResult,
    search_documents
)

__all__ = [
    'QdrantVectorStore',
    'SearchResult',
    'search_documents'
]
