"""
Document ingestion and processing module.

This module handles:
- Document loading (PDF, DOCX, TXT, etc.)
- Conversion to Markdown
- Image extraction and processing
- Text preprocessing
- Semantic chunking
- Research paper metadata extraction
- Storage in SQLite

WHY THE EXPORTS ARE LAZY
------------------------
Importing a submodule runs this file first, so `from docusense.ingestion.chunker
import Chunk` — which the retrieval path does — used to load the document
converter as well, and with it markitdown, magika (an ONNX file-type
classifier), pandas and PIL. Measured: a process that only ever answers
questions was carrying the whole conversion stack, and answering one query took
600MB instead of 456MB.

PEP 562 keeps every name importable from the package while loading it on first
use, so the query path pays for the query path.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # noqa: F401 - for type checkers only; never imported at runtime
    from docusense.ingestion.chunker import Chunk, SemanticChunker  # noqa: F401
    from docusense.ingestion.converters import ConversionResult, DocumentConverter  # noqa: F401
    from docusense.ingestion.image_processor import (
        ImageDescription,  # noqa: F401
        ImageProcessor,  # noqa: F401
        VisionProvider,  # noqa: F401
    )
    from docusense.ingestion.paper_metadata import (
        Citation,  # noqa: F401
        PaperMetadata,  # noqa: F401
        PaperMetadataExtractor,  # noqa: F401
        PaperSection,  # noqa: F401
        extract_paper_metadata,  # noqa: F401
    )
    from docusense.ingestion.pipeline import (
        DocumentPipeline,  # noqa: F401
        PipelineResult,  # noqa: F401
        process_document,  # noqa: F401
    )
    from docusense.ingestion.preprocessor import (
        PreprocessResult,  # noqa: F401
        TextPreprocessor,  # noqa: F401
        preprocess_text,  # noqa: F401
    )


# name -> submodule it lives in
_EXPORTS = {
    "Chunk": "chunker",
    "SemanticChunker": "chunker",
    "ConversionResult": "converters",
    "DocumentConverter": "converters",
    "ImageDescription": "image_processor",
    "ImageProcessor": "image_processor",
    "VisionProvider": "image_processor",
    "Citation": "paper_metadata",
    "PaperMetadata": "paper_metadata",
    "PaperMetadataExtractor": "paper_metadata",
    "PaperSection": "paper_metadata",
    "extract_paper_metadata": "paper_metadata",
    "DocumentPipeline": "pipeline",
    "PipelineResult": "pipeline",
    "process_document": "pipeline",
    "PreprocessResult": "preprocessor",
    "TextPreprocessor": "preprocessor",
    "preprocess_text": "preprocessor",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import the submodule that owns `name`, the first time it is asked for."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # cached, so this runs once per name
    return value


def __dir__():
    return sorted(set(__all__) | set(globals()))
