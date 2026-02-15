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

from docusense.ingestion.converters import DocumentConverter, ConversionResult
from docusense.ingestion.image_processor import ImageProcessor, ImageDescription, VisionProvider
from docusense.ingestion.preprocessor import TextPreprocessor, PreprocessResult, preprocess_text
from docusense.ingestion.chunker import SemanticChunker, Chunk
from docusense.ingestion.pipeline import DocumentPipeline, PipelineResult, process_document

__all__ = [
    "DocumentConverter",
    "ConversionResult",
    "ImageProcessor",
    "ImageDescription",
    "VisionProvider",
    "TextPreprocessor",
    "PreprocessResult",
    "preprocess_text",
    "SemanticChunker",
    "Chunk",
    "DocumentPipeline",
    "PipelineResult",
    "process_document",
]
