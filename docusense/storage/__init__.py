"""
Storage layer for DocuSense.

Exports:
- ChunkStorage: SQLite storage manager
- DocumentRecord, ChunkRecord, ImageRecord: Database models
- get_storage: Convenience function for storage access
"""

from .chunk_store import (
    ChunkStorage,
    DocumentRecord,
    ChunkRecord,
    ImageRecord,
    get_storage
)

__all__ = [
    'ChunkStorage',
    'DocumentRecord',
    'ChunkRecord',
    'ImageRecord',
    'get_storage'
]
