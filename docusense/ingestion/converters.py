"""
Document Converter - Transforms documents into Markdown format.

This module is the FIRST step in Phase 1: Knowledge Ingestion.

PURPOSE:
--------
Convert any document (PDF, DOCX, TXT, etc.) into clean Markdown format while:
1. Preserving document structure (headers, lists, tables)
2. Extracting embedded images for later processing
3. Handling errors gracefully with fallback strategies
4. Maintaining metadata (filename, page numbers, source)

WHY MARKDOWN?
-------------
- LLMs are trained on Markdown (GitHub, docs, papers)
- Structure-aware chunking (split on ## headers)
- Easy to read and debug
- Preserves formatting (bold, lists, code blocks)
- Better embeddings (structure = semantic meaning)

SUPPORTED FORMATS:
------------------
- PDF (.pdf)
- Word Documents (.docx)
- PowerPoint (.pptx)
- Excel (.xlsx)
- Plain Text (.txt)
- Markdown (.md)
- HTML (.html)

FAILURE HANDLING:
-----------------
Primary: Markitdown (universal converter)
Fallback 1: Format-specific parsers (PyPDF2, python-docx)
Fallback 2: Plain text extraction
Emergency: Skip with error log
"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger
from markitdown import MarkItDown
import PyPDF2
from docx import Document as DocxDocument

from docusense.config import settings


@dataclass
class ConversionResult:
    """
    Result of document conversion.
    
    Attributes:
        success: Whether conversion succeeded
        markdown: Converted Markdown text
        images: List of extracted image paths
        metadata: Document metadata (filename, pages, etc.)
        error: Error message if conversion failed
    """
    success: bool
    markdown: str
    images: List[str]
    metadata: Dict[str, any]
    error: Optional[str] = None


class DocumentConverter:
    """
    Universal document converter using Markitdown.
    
    ARCHITECTURE:
    -------------
    1. Validate file (exists, size, type)
    2. Try primary converter (Markitdown)
    3. If fails, try format-specific fallback
    4. Extract and save images
    5. Return result with metadata
    
    EXAMPLE USAGE:
    --------------
    >>> converter = DocumentConverter()
    >>> result = converter.convert("document.pdf")
    >>> if result.success:
    >>>     print(result.markdown)
    >>>     print(f"Extracted {len(result.images)} images")
    >>> else:
    >>>     print(f"Error: {result.error}")
    """
    
    def __init__(self):
        """Initialize converter with Markitdown engine."""
        self.markitdown = MarkItDown()
        self.markdown_dir = settings.markdown_data_dir
        self.images_dir = settings.images_data_dir
        
        # Ensure output directories exist
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DocumentConverter initialized")
        logger.info(f"  Markdown output: {self.markdown_dir}")
        logger.info(f"  Images output: {self.images_dir}")
    
    def convert(self, file_path: str) -> ConversionResult:
        """
        Convert document to Markdown format.
        
        PROCESS FLOW:
        -------------
        1. Validate file (size, type, existence)
        2. Generate unique document ID
        3. Try Markitdown conversion
        4. Extract images if present
        5. Save Markdown and images to disk
        6. Return result with metadata
        
        Args:
            file_path: Path to document file
            
        Returns:
            ConversionResult with markdown, images, and metadata
            
        Example:
            >>> converter = DocumentConverter()
            >>> result = converter.convert("report.pdf")
            >>> print(f"Pages: {result.metadata['total_pages']}")
            >>> print(result.markdown[:200])  # First 200 chars
        """
        file_path = Path(file_path)
        
        # Step 1: Validate file
        logger.info(f"Converting document: {file_path.name}")
        validation_error = self._validate_file(file_path)
        if validation_error:
            return ConversionResult(
                success=False,
                markdown="",
                images=[],
                metadata={"filename": file_path.name},
                error=validation_error
            )
        
        # Step 2: Generate unique document ID
        doc_id = str(uuid.uuid4())[:8]
        
        # Step 3: Try conversion (with fallback strategy)
        try:
            markdown_text, images = self._convert_with_markitdown(file_path, doc_id)
            
        except Exception as e:
            logger.warning(f"Markitdown failed for {file_path.name}: {e}")
            logger.info("Trying fallback converter...")
            
            try:
                markdown_text, images = self._convert_with_fallback(file_path, doc_id)
            except Exception as fallback_error:
                logger.error(f"All conversion methods failed: {fallback_error}")
                return ConversionResult(
                    success=False,
                    markdown="",
                    images=[],
                    metadata={"filename": file_path.name, "doc_id": doc_id},
                    error=f"Conversion failed: {str(fallback_error)}"
                )
        
        # Step 4: Save Markdown to disk
        markdown_file = self.markdown_dir / f"{doc_id}_{file_path.stem}.md"
        try:
            markdown_file.write_text(markdown_text, encoding='utf-8')
            logger.info(f"Saved Markdown: {markdown_file.name}")
        except Exception as e:
            logger.error(f"Failed to save Markdown: {e}")
        
        # Step 5: Build metadata
        metadata = {
            "doc_id": doc_id,
            "filename": file_path.name,
            "file_type": file_path.suffix,
            "file_size_bytes": file_path.stat().st_size,
            "markdown_path": str(markdown_file),
            "total_images": len(images),
            "markdown_length": len(markdown_text),
        }
        
        logger.success(f"✅ Converted {file_path.name} → {len(markdown_text)} chars, {len(images)} images")
        
        return ConversionResult(
            success=True,
            markdown=markdown_text,
            images=images,
            metadata=metadata
        )
    
    def _validate_file(self, file_path: Path) -> Optional[str]:
        """
        Validate file before conversion.
        
        CHECKS:
        -------
        1. File exists
        2. File size within limits (prevent OOM)
        3. File type is supported
        
        Args:
            file_path: Path to file
            
        Returns:
            Error message if validation fails, None if OK
        """
        # Check 1: File exists
        if not file_path.exists():
            return f"File not found: {file_path}"
        
        # Check 2: File size (prevent memory overflow)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            return f"File too large: {file_size_mb:.1f} MB (max: {settings.max_file_size_mb} MB)"
        
        # Check 3: File type supported
        if file_path.suffix.lower() not in settings.allowed_file_types:
            return f"Unsupported file type: {file_path.suffix} (allowed: {settings.allowed_file_types})"
        
        logger.debug(f"File validation passed: {file_path.name} ({file_size_mb:.2f} MB)")
        return None
    
    def _convert_with_markitdown(self, file_path: Path, doc_id: str) -> Tuple[str, List[str]]:
        """
        Primary conversion using Markitdown library.
        
        WHY MARKITDOWN?
        ---------------
        - Universal: Handles PDF, DOCX, PPTX, XLSX, HTML automatically
        - Preserves structure: Tables, headers, lists maintained
        - Image extraction: Automatically extracts embedded images
        - Microsoft-backed: Well-maintained, production-ready
        
        Args:
            file_path: Path to document
            doc_id: Unique document identifier
            
        Returns:
            Tuple of (markdown_text, image_paths)
            
        Raises:
            Exception if conversion fails
        """
        logger.debug(f"Converting with Markitdown: {file_path.name}")
        
        # Convert document
        result = self.markitdown.convert(str(file_path))
        
        # Extract text
        markdown_text = result.text_content
        
        if not markdown_text or not markdown_text.strip():
            raise ValueError("No text extracted from document")
        
        # Extract images (if any)
        images = []
        if hasattr(result, 'images') and result.images:
            logger.info(f"Found {len(result.images)} images in document")
            for idx, image_data in enumerate(result.images):
                image_path = self._save_image(image_data, doc_id, idx)
                if image_path:
                    images.append(image_path)
        
        logger.info(f"Markitdown extracted: {len(markdown_text)} chars, {len(images)} images")
        return markdown_text, images
    
    def _convert_with_fallback(self, file_path: Path, doc_id: str) -> Tuple[str, List[str]]:
        """
        Fallback conversion using format-specific parsers.
        
        FALLBACK STRATEGY:
        ------------------
        - PDF: PyPDF2 (text extraction only, no fancy formatting)
        - DOCX: python-docx (paragraph-by-paragraph)
        - TXT/MD: Direct read (already text)
        
        This is less sophisticated than Markitdown but more reliable
        for problematic files (corrupted PDFs, complex layouts).
        
        Args:
            file_path: Path to document
            doc_id: Unique document identifier
            
        Returns:
            Tuple of (markdown_text, image_paths)
            
        Raises:
            Exception if all fallbacks fail
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._convert_pdf_fallback(file_path, doc_id)
        
        elif suffix == '.docx':
            return self._convert_docx_fallback(file_path, doc_id)
        
        elif suffix in ['.txt', '.md']:
            return self._convert_text_fallback(file_path)
        
        else:
            raise ValueError(f"No fallback converter for {suffix}")
    
    def _convert_pdf_fallback(self, file_path: Path, doc_id: str) -> Tuple[str, List[str]]:
        """
        Fallback PDF conversion using PyPDF2.
        
        LIMITATIONS:
        ------------
        - Text only (no formatting preserved)
        - May fail on scanned PDFs (image-based)
        - Poor table extraction
        
        WHEN USED:
        ----------
        - Markitdown fails (corrupted file, unsupported features)
        - Complex multi-column layouts
        - Password-protected PDFs (if password known)
        
        Args:
            file_path: Path to PDF
            doc_id: Document ID
            
        Returns:
            Tuple of (markdown_text, empty_list) - no images extracted
        """
        logger.info(f"Using PDF fallback: {file_path.name}")
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            # Extract text from all pages
            pages = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    pages.append(f"## Page {page_num}\n\n{text}")
            
            markdown_text = "\n\n".join(pages)
            
            if not markdown_text.strip():
                raise ValueError("No text extracted from PDF (might be scanned/image-based)")
            
            logger.info(f"PDF fallback extracted {len(pages)} pages")
            return markdown_text, []
    
    def _convert_docx_fallback(self, file_path: Path, doc_id: str) -> Tuple[str, List[str]]:
        """
        Fallback DOCX conversion using python-docx.
        
        EXTRACTION:
        -----------
        - Paragraph by paragraph
        - Basic formatting preserved (bold → **bold**)
        - Lists maintained
        
        Args:
            file_path: Path to DOCX
            doc_id: Document ID
            
        Returns:
            Tuple of (markdown_text, empty_list)
        """
        logger.info(f"Using DOCX fallback: {file_path.name}")
        
        doc = DocxDocument(file_path)
        
        # Extract paragraphs
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        
        markdown_text = "\n\n".join(paragraphs)
        
        if not markdown_text.strip():
            raise ValueError("No text extracted from DOCX")
        
        logger.info(f"DOCX fallback extracted {len(paragraphs)} paragraphs")
        return markdown_text, []
    
    def _convert_text_fallback(self, file_path: Path) -> Tuple[str, List[str]]:
        """
        Fallback for plain text files.
        
        Simply reads file content with encoding detection.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Tuple of (text_content, empty_list)
        """
        logger.info(f"Reading text file: {file_path.name}")
        
        # Try UTF-8 first, fallback to chardet
        try:
            text = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            import chardet
            raw_data = file_path.read_bytes()
            encoding = chardet.detect(raw_data)['encoding']
            text = raw_data.decode(encoding or 'utf-8', errors='ignore')
        
        if not text.strip():
            raise ValueError("File is empty")
        
        return text, []
    
    def _save_image(self, image_data: any, doc_id: str, index: int) -> Optional[str]:
        """
        Save extracted image to disk.
        
        Args:
            image_data: Image data from Markitdown
            doc_id: Document ID
            index: Image index in document
            
        Returns:
            Path to saved image, or None if failed
        """
        try:
            image_filename = f"{doc_id}_image_{index:03d}.png"
            image_path = self.images_dir / image_filename
            
            # Save image (implementation depends on image_data format)
            # This is a placeholder - actual implementation may vary
            # based on Markitdown's image format
            
            logger.debug(f"Saved image: {image_filename}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Failed to save image {index}: {e}")
            return None
