"""
Document ingestion and processing module.

This module handles:
- Document loading (PDF, DOCX, TXT, etc.)
- Conversion to Markdown
- Image extraction and processing
- Text preprocessing
- Semantic chunking
- Research paper metadata extraction (NEW)
- Storage in SQLite
"""

from docusense.ingestion.converters import DocumentConverter, ConversionResult
from docusense.ingestion.image_processor import ImageProcessor, ImageDescription, VisionProvider
from docusense.ingestion.preprocessor import TextPreprocessor, PreprocessResult, preprocess_text
from docusense.ingestion.chunker import SemanticChunker, Chunk
from docusense.ingestion.paper_metadata import (
    PaperMetadataExtractor, 
    PaperMetadata, 
    PaperSection,
    Citation,
    extract_paper_metadata
)
from docusense.ingestion.pipeline import DocumentPipeline, PipelineResult, process_document

__all__ = [
    "Chunk",
    "Citation",
    "ConversionResult",
    "DocumentConverter",
    "DocumentPipeline",
    "ImageDescription",
    "ImageProcessor",
    "PaperMetadata",
    "PaperMetadataExtractor",
    "PaperSection",
    "PipelineResult",
    "PreprocessResult",
    "SemanticChunker",
    "TextPreprocessor",
    "VisionProvider",
    "extract_paper_metadata",
    "preprocess_text",
    "process_document",
]
