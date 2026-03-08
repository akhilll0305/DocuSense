"""
Storage layer for DocuSense.

Exports:
- ChunkStorage: SQLite storage manager
- DocumentRecord, ChunkRecord, ImageRecord: Database models
- ConversationStore: Conversation and query history storage
- get_storage: Convenience function for storage access
"""

from .chunk_store import (
    ChunkStorage,
    DocumentRecord,
    ChunkRecord,
    ImageRecord,
    get_storage
)

from .conversation_store import (
    ConversationStore,
    Conversation,
    Message,
    QueryLog
)

__all__ = [
    'ChunkStorage',
    'DocumentRecord',
    'ChunkRecord',
    'ImageRecord',
    'get_storage',
    'ConversationStore',
    'Conversation',
    'Message',
    'QueryLog',
]

