"""
Document ingestion and processing module.

This module handles:
- Document loading (PDF, DOCX, TXT, etc.)
- Conversion to Markdown
- Image extraction and processing
- Text preprocessing
- Semantic chunking
- Storage in SQLite
"""

from docusense.ingestion.converters import DocumentConverter

__all__ = ["DocumentConverter"]
