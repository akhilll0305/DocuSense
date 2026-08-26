"""
SQLite storage layer for document chunks and metadata.

This module provides persistent storage for:
1. Documents (files ingested into the system)
2. Chunks (semantic pieces of documents)
3. Images (extracted from documents with descriptions)

Database Schema:
- documents: Metadata about ingested files
- chunks: Semantic chunks with text and metadata
- images: Images extracted from documents with vision descriptions

Author: DocuSense
Created: 2025
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger

from docusense.config.settings import settings


@dataclass
class DocumentRecord:
    """Database record for a document."""
    document_id: str
    filename: str
    file_path: str
    file_type: str  # pdf, docx, txt, etc.
    total_chunks: int
    processing_date: str
    metadata: Dict[str, Any]  # Flexible JSON metadata
    
    # Optional fields (set by database on insert)
    id: Optional[int] = None
    

@dataclass
class ChunkRecord:
    """Database record for a chunk."""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    
    # Structural metadata
    header_path: str = ""
    page_number: Optional[int] = None
    
    # Content flags
    has_code: bool = False
    has_tables: bool = False
    has_overlap: bool = False
    merged: bool = False
    emergency_split: bool = False
    
    # Additional metadata as JSON
    metadata: Dict[str, Any] = None
    
    # Optional fields (set by database on insert)
    id: Optional[int] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Ensure metadata is a dict."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ImageRecord:
    """Database record for an image extracted from a document."""
    image_id: str
    document_id: str
    image_path: str
    description: str  # From vision model
    ocr_text: Optional[str] = None  # From OCR fallback
    vision_provider: Optional[str] = None  # gemini, llava, ocr
    metadata: Dict[str, Any] = None
    
    # Optional fields (set by database on insert)
    id: Optional[int] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Ensure metadata is a dict."""
        if self.metadata is None:
            self.metadata = {}


class ChunkStorage:
    """
    SQLite storage manager for document chunks and metadata.
    
    Features:
    - Transactional safety (ACID compliance)
    - Foreign key constraints (referential integrity)
    - Efficient indexing for fast queries
    - JSON metadata support
    
    Usage:
        storage = ChunkStorage()
        storage.create_schema()
        
        # Add a document
        doc_id = storage.add_document(document_record)
        
        # Add chunks
        for chunk in chunks:
            storage.add_chunk(chunk_record)
        
        # Query chunks
        chunks = storage.get_chunks_by_document(doc_id)
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize storage connection.
        
        Args:
            db_path: Path to SQLite database file (default from settings)
        """
        self.db_path = db_path or settings.sqlite_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChunkStorage at {self.db_path}")
        
        # Create connection (will be reused)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        
        # Enable foreign keys (critical for referential integrity!)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        logger.success(f"✅ Connected to database: {self.db_path}")
    
    def create_schema(self):
        """
        Create database tables if they don't exist.
        
        Tables:
        1. documents: File metadata
        2. chunks: Semantic chunks with text
        3. images: Extracted images with descriptions
        
        Indexes:
        - document_id on chunks (fast chunk retrieval)
        - document_id on images (fast image retrieval)
        - chunk_id (unique identifier lookup)
        """
        logger.info("Creating database schema...")
        
        cursor = self.conn.cursor()
        
        # 1. DOCUMENTS TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                total_chunks INTEGER NOT NULL DEFAULT 0,
                processing_date TEXT NOT NULL,
                metadata TEXT,  -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. CHUNKS TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                header_path TEXT,
                page_number INTEGER,
                has_code BOOLEAN DEFAULT 0,
                has_tables BOOLEAN DEFAULT 0,
                has_overlap BOOLEAN DEFAULT 0,
                merged BOOLEAN DEFAULT 0,
                emergency_split BOOLEAN DEFAULT 0,
                metadata TEXT,  -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            )
        """)
        
        # 3. IMAGES TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                description TEXT,
                ocr_text TEXT,
                vision_provider TEXT,
                metadata TEXT,  -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            )
        """)
        
        # 4. CREATE INDEXES for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
            ON chunks(document_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id 
            ON chunks(chunk_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_document_id 
            ON images(document_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_document_id 
            ON documents(document_id)
        """)
        
        self.conn.commit()
        logger.success("✅ Database schema created successfully")
        
        # Log table stats
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        logger.info(f"Tables: {', '.join(t[0] for t in tables)}")
    
    def add_document(self, doc: DocumentRecord) -> int:
        """
        Add a document record to the database.
        
        Args:
            doc: DocumentRecord to insert
        
        Returns:
            Row ID of inserted document
        
        Raises:
            sqlite3.IntegrityError: If document_id already exists
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents (
                document_id, filename, file_path, file_type,
                total_chunks, processing_date, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.document_id,
            doc.filename,
            doc.file_path,
            doc.file_type,
            doc.total_chunks,
            doc.processing_date,
            json.dumps(doc.metadata)
        ))
        
        self.conn.commit()
        row_id = cursor.lastrowid
        
        logger.info(f"Added document {doc.document_id}: {doc.filename} ({doc.total_chunks} chunks)")
        return row_id
    
    def add_chunk(self, chunk: ChunkRecord) -> int:
        """
        Add a chunk record to the database.
        
        Args:
            chunk: ChunkRecord to insert
        
        Returns:
            Row ID of inserted chunk
        
        Raises:
            sqlite3.IntegrityError: If chunk_id already exists or document_id invalid
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO chunks (
                chunk_id, document_id, chunk_index, text, token_count,
                header_path, page_number, has_code, has_tables, has_overlap,
                merged, emergency_split, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk.chunk_id,
            chunk.document_id,
            chunk.chunk_index,
            chunk.text,
            chunk.token_count,
            chunk.header_path,
            chunk.page_number,
            1 if chunk.has_code else 0,
            1 if chunk.has_tables else 0,
            1 if chunk.has_overlap else 0,
            1 if chunk.merged else 0,
            1 if chunk.emergency_split else 0,
            json.dumps(chunk.metadata)
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def add_chunks_bulk(self, chunks: List[ChunkRecord]) -> int:
        """
        Add multiple chunks in a single transaction (much faster!).
        
        Args:
            chunks: List of ChunkRecords to insert
        
        Returns:
            Number of chunks inserted
        """
        if not chunks:
            return 0
        
        cursor = self.conn.cursor()
        
        data = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.text,
                chunk.token_count,
                chunk.header_path,
                chunk.page_number,
                1 if chunk.has_code else 0,
                1 if chunk.has_tables else 0,
                1 if chunk.has_overlap else 0,
                1 if chunk.merged else 0,
                1 if chunk.emergency_split else 0,
                json.dumps(chunk.metadata)
            )
            for chunk in chunks
        ]
        
        cursor.executemany("""
            INSERT INTO chunks (
                chunk_id, document_id, chunk_index, text, token_count,
                header_path, page_number, has_code, has_tables, has_overlap,
                merged, emergency_split, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        self.conn.commit()
        
        logger.info(f"Bulk inserted {len(chunks)} chunks for document {chunks[0].document_id}")
        return len(chunks)
    
    def add_image(self, image: ImageRecord) -> int:
        """
        Add an image record to the database.
        
        Args:
            image: ImageRecord to insert
        
        Returns:
            Row ID of inserted image
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO images (
                image_id, document_id, image_path, description,
                ocr_text, vision_provider, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            image.image_id,
            image.document_id,
            image.image_path,
            image.description,
            image.ocr_text,
            image.vision_provider,
            json.dumps(image.metadata)
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Unique document identifier
        
        Returns:
            DocumentRecord or None if not found
        """
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return DocumentRecord(
            id=row['id'],
            document_id=row['document_id'],
            filename=row['filename'],
            file_path=row['file_path'],
            file_type=row['file_type'],
            total_chunks=row['total_chunks'],
            processing_date=row['processing_date'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    @staticmethod
    def _row_to_chunk(row) -> ChunkRecord:
        """Map a chunks table row to a ChunkRecord."""
        return ChunkRecord(
            id=row['id'],
            chunk_id=row['chunk_id'],
            document_id=row['document_id'],
            chunk_index=row['chunk_index'],
            text=row['text'],
            token_count=row['token_count'],
            header_path=row['header_path'] or "",
            page_number=row['page_number'],
            has_code=bool(row['has_code']),
            has_tables=bool(row['has_tables']),
            has_overlap=bool(row['has_overlap']),
            merged=bool(row['merged']),
            emergency_split=bool(row['emergency_split']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=row['created_at']
        )

    def get_chunks_by_document(self, document_id: str) -> List[ChunkRecord]:
        """
        Retrieve all chunks for a document, ordered by chunk_index.

        Args:
            document_id: Unique document identifier

        Returns:
            List of ChunkRecords (ordered by chunk_index)
        """
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,)
        ).fetchall()

        return [self._row_to_chunk(row) for row in rows]

    def get_all_chunks(self) -> List[ChunkRecord]:
        """
        Retrieve every chunk in the database, ordered by document then index.

        Used to build the in-memory BM25 corpus, which needs the full text of
        all chunks rather than the vector store's payloads.

        Returns:
            List of ChunkRecords across all documents
        """
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM chunks ORDER BY document_id, chunk_index"
        ).fetchall()

        return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """
        Retrieve a single chunk by ID.
        
        Args:
            chunk_id: Unique chunk identifier
        
        Returns:
            ChunkRecord or None if not found
        """
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?",
            (chunk_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return ChunkRecord(
            id=row['id'],
            chunk_id=row['chunk_id'],
            document_id=row['document_id'],
            chunk_index=row['chunk_index'],
            text=row['text'],
            token_count=row['token_count'],
            header_path=row['header_path'] or "",
            page_number=row['page_number'],
            has_code=bool(row['has_code']),
            has_tables=bool(row['has_tables']),
            has_overlap=bool(row['has_overlap']),
            merged=bool(row['merged']),
            emergency_split=bool(row['emergency_split']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=row['created_at']
        )
    
    def get_images_by_document(self, document_id: str) -> List[ImageRecord]:
        """
        Retrieve all images for a document.
        
        Args:
            document_id: Unique document identifier
        
        Returns:
            List of ImageRecords
        """
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM images WHERE document_id = ?",
            (document_id,)
        ).fetchall()
        
        return [
            ImageRecord(
                id=row['id'],
                image_id=row['image_id'],
                document_id=row['document_id'],
                image_path=row['image_path'],
                description=row['description'],
                ocr_text=row['ocr_text'],
                vision_provider=row['vision_provider'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                created_at=row['created_at']
            )
            for row in rows
        ]
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all associated chunks/images (CASCADE).
        
        Args:
            document_id: Unique document identifier
        
        Returns:
            True if deleted, False if document not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM documents WHERE document_id = ?",
            (document_id,)
        )
        self.conn.commit()
        
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted document {document_id} and associated chunks/images")
        return deleted
    
    def get_all_documents(self) -> List[DocumentRecord]:
        """
        Retrieve all documents in the database.
        
        Returns:
            List of DocumentRecords
        """
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
        
        return [
            DocumentRecord(
                id=row['id'],
                document_id=row['document_id'],
                filename=row['filename'],
                file_path=row['file_path'],
                file_type=row['file_type'],
                total_chunks=row['total_chunks'],
                processing_date=row['processing_date'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            for row in rows
        ]
    
    def count_chunks(self, document_id: Optional[str] = None) -> int:
        """
        Count total chunks (optionally for a specific document).
        
        Args:
            document_id: Optional document filter
        
        Returns:
            Number of chunks
        """
        cursor = self.conn.cursor()
        
        if document_id:
            result = cursor.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                (document_id,)
            ).fetchone()
        else:
            result = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()
        
        return result[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict with document/chunk/image counts and averages
        """
        cursor = self.conn.cursor()
        
        doc_count = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        image_count = cursor.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        
        avg_chunks = 0
        avg_tokens = 0
        
        if chunk_count > 0:
            avg_chunks = chunk_count / max(doc_count, 1)
            avg_tokens = cursor.execute(
                "SELECT AVG(token_count) FROM chunks"
            ).fetchone()[0]
        
        return {
            'documents': doc_count,
            'chunks': chunk_count,
            'images': image_count,
            'avg_chunks_per_doc': round(avg_chunks, 1),
            'avg_tokens_per_chunk': round(avg_tokens, 1) if avg_tokens else 0
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()


# Convenience function for quick storage access
def get_storage(db_path: Optional[Path] = None) -> ChunkStorage:
    """
    Get a ChunkStorage instance with schema initialized.
    
    Args:
        db_path: Optional path to database file
    
    Returns:
        ChunkStorage instance (caller responsible for closing)
    """
    storage = ChunkStorage(db_path)
    storage.create_schema()
    return storage
