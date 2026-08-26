"""
End-to-end document ingestion pipeline.

This module orchestrates all Phase 1 components:
1. Document conversion (any format → Markdown)
2. Image processing (vision models → descriptions)
3. Text preprocessing (clean while preserving structure)
4. Semantic chunking (split into meaningful pieces)
5. Storage (persist to SQLite database)

Author: DocuSense
Created: 2025
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from tqdm import tqdm

from docusense.ingestion.converters import DocumentConverter, ConversionResult
from docusense.ingestion.image_processor import ImageProcessor, ImageDescription
from docusense.ingestion.preprocessor import TextPreprocessor
from docusense.ingestion.chunker import SemanticChunker, Chunk
from docusense.ingestion.paper_metadata import (
    PaperMetadataExtractor, 
    PaperMetadata
)
from docusense.storage import (
    ChunkStorage,
    DocumentRecord,
    ChunkRecord,
    ImageRecord,
    get_storage
)


@dataclass
class PipelineResult:
    """Result of running a document through the pipeline."""
    success: bool
    document_id: str
    filename: str
    file_path: str
    
    # Processing stats
    total_chunks: int = 0
    total_images: int = 0
    processing_time_seconds: float = 0.0
    
    # Component results
    conversion_result: Optional[ConversionResult] = None
    chunks: List[Chunk] = field(default_factory=list)
    images: List[ImageDescription] = field(default_factory=list)
    paper_metadata: Optional[PaperMetadata] = None  # NEW: Research paper metadata
    
    # Error info
    error: Optional[str] = None
    error_stage: Optional[str] = None
    
    def __str__(self) -> str:
        """Human-readable summary."""
        if self.success:
            paper_info = ""
            if self.paper_metadata and self.paper_metadata.is_research_paper():
                paper_info = f" [Research Paper: {self.paper_metadata.confidence:.1%}]"
            return (
                f"✅ {self.filename}: {self.total_chunks} chunks, "
                f"{self.total_images} images ({self.processing_time_seconds:.1f}s){paper_info}"
            )
        else:
            return f"❌ {self.filename}: Failed at {self.error_stage} - {self.error}"


class DocumentPipeline:
    """
    End-to-end pipeline for document ingestion.
    
    Features:
    - Multi-format support (PDF, DOCX, TXT, etc.)
    - Image understanding via vision models
    - Semantic chunking with context preservation
    - Persistent storage in SQLite
    - Progress tracking with tqdm
    - Comprehensive error handling
    
    Usage:
        pipeline = DocumentPipeline()
        result = pipeline.process_document("path/to/document.pdf")
        
        if result.success:
            print(f"✅ Processed {result.total_chunks} chunks")
        else:
            print(f"❌ Error: {result.error}")
    
    Batch processing:
        results = pipeline.process_batch([
            "doc1.pdf",
            "doc2.docx",
            "doc3.txt"
        ])
    """
    
    def __init__(
        self,
        storage: Optional[ChunkStorage] = None,
        converter: Optional[DocumentConverter] = None,
        image_processor: Optional[ImageProcessor] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        chunker: Optional[SemanticChunker] = None,
        paper_metadata_extractor: Optional[PaperMetadataExtractor] = None,  # NEW
        enable_images: bool = True,
        enable_paper_extraction: bool = True  # NEW
    ):
        """
        Initialize the pipeline with all components.
        
        Args:
            storage: ChunkStorage instance (creates new if None)
            converter: DocumentConverter instance (creates new if None)
            image_processor: ImageProcessor instance (creates new if None)
            preprocessor: TextPreprocessor instance (creates new if None)
            chunker: SemanticChunker instance (creates new if None)
            paper_metadata_extractor: PaperMetadataExtractor instance (creates new if None)
            enable_images: Whether to process images (default True)
            enable_paper_extraction: Whether to extract research paper metadata (default True)
        """
        logger.info("Initializing DocumentPipeline...")
        
        # Initialize components (create defaults if not provided)
        self.storage = storage or get_storage()
        self.converter = converter or DocumentConverter()
        self.image_processor = image_processor or ImageProcessor() if enable_images else None
        self.preprocessor = preprocessor or TextPreprocessor()
        self.chunker = chunker or SemanticChunker()
        self.paper_metadata_extractor = paper_metadata_extractor or PaperMetadataExtractor() if enable_paper_extraction else None
        
        self.enable_images = enable_images
        self.enable_paper_extraction = enable_paper_extraction
        
        logger.success("✅ DocumentPipeline initialized")
        logger.info("  Components: Converter, Preprocessor, Chunker, Storage")
        logger.info(f"  Image processing: {'Enabled' if enable_images else 'Disabled'}")
        logger.info(f"  Paper metadata extraction: {'Enabled' if enable_paper_extraction else 'Disabled'}")
    
    def process_document(
        self,
        file_path: str | Path,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Process a single document through the entire pipeline.
        
        Steps:
        1. Convert document to Markdown
        2. Process images (if any)
        3. Preprocess text
        4. Chunk into semantic pieces
        5. Store in database
        
        Args:
            file_path: Path to document file
            document_id: Optional custom document ID (auto-generated if None)
            metadata: Optional metadata dict to attach to document
        
        Returns:
            PipelineResult with success status and details
        """
        file_path = Path(file_path)
        start_time = datetime.now()
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = self._generate_document_id(file_path)
        
        logger.info(f"🚀 Processing document: {file_path.name}")
        logger.info(f"   Document ID: {document_id}")
        
        result = PipelineResult(
            success=False,
            document_id=document_id,
            filename=file_path.name,
            file_path=str(file_path)
        )
        
        try:
            # ================================================================
            # STAGE 1: DOCUMENT CONVERSION
            # ================================================================
            logger.info("📄 Stage 1/5: Converting document to Markdown...")
            conversion_result = self.converter.convert(str(file_path))
            
            if not conversion_result.success:
                result.error = f"Conversion failed: {conversion_result.error}"
                result.error_stage = "conversion"
                logger.error(f"❌ {result.error}")
                return result
            
            result.conversion_result = conversion_result
            logger.success(f"✅ Converted to Markdown ({len(conversion_result.markdown)} chars)")
            
            # ================================================================
            # STAGE 2: IMAGE PROCESSING
            # ================================================================
            image_descriptions = []
            if self.enable_images and self.image_processor and conversion_result.images:
                logger.info(f"🖼️ Stage 2/5: Processing {len(conversion_result.images)} images...")
                
                for img_path in tqdm(conversion_result.images, desc="Processing images", unit="img"):
                    try:
                        description = self.image_processor.process_image(
                            str(img_path),
                            context=f"Image from {file_path.name}"
                        )
                        if description:
                            image_descriptions.append(description)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process image {img_path}: {e}")
                
                result.images = image_descriptions
                result.total_images = len(image_descriptions)
                logger.success(f"✅ Processed {len(image_descriptions)} images")
            else:
                logger.info("⏭️ Stage 2/5: Skipping image processing (disabled or no images)")
            
            # ================================================================
            # STAGE 2.5: RESEARCH PAPER METADATA EXTRACTION (NEW!)
            # ================================================================
            paper_metadata = None
            if self.enable_paper_extraction and self.paper_metadata_extractor:
                logger.info("📚 Stage 2.5/5: Extracting research paper metadata...")
                try:
                    paper_metadata = self.paper_metadata_extractor.extract_from_markdown(
                        conversion_result.markdown,
                        file_path if file_path.suffix.lower() == '.pdf' else None
                    )
                    result.paper_metadata = paper_metadata
                    
                    if paper_metadata.is_research_paper():
                        logger.success(
                            f"✅ Research paper detected (confidence: {paper_metadata.confidence:.1%})"
                        )
                        logger.info(f"   Title: {paper_metadata.title[:60] if paper_metadata.title else 'Unknown'}...")
                        logger.info(f"   Authors: {len(paper_metadata.authors)} detected")
                        logger.info(f"   Year: {paper_metadata.year}")
                        logger.info(f"   Sections: {len(paper_metadata.sections)}")
                    else:
                        logger.info(
                            f"⏭️ Not a research paper (confidence: {paper_metadata.confidence:.1%}), "
                            "treating as generic document"
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Paper metadata extraction failed: {e}")
                    paper_metadata = None
            else:
                logger.info("⏭️ Stage 2.5/5: Skipping paper metadata extraction (disabled)")
            
            # ================================================================
            # STAGE 3: TEXT PREPROCESSING
            # ================================================================
            logger.info("🧹 Stage 3/5: Preprocessing text...")
            preprocess_result = self.preprocessor.process(conversion_result.markdown)
            
            cleaned_text = preprocess_result.cleaned_text
            chars_removed = len(preprocess_result.original_text) - len(preprocess_result.cleaned_text)
            reduction = (chars_removed / len(preprocess_result.original_text) * 100) if preprocess_result.original_text else 0
            logger.success(
                f"✅ Preprocessed text ({chars_removed} chars removed, "
                f"{reduction:.1f}% reduction)"
            )
            
            # ================================================================
            # STAGE 4: SEMANTIC CHUNKING
            # ================================================================
            logger.info("✂️ Stage 4/5: Chunking into semantic pieces...")
            chunks = self.chunker.chunk(cleaned_text, doc_id=document_id)
            
            # Enrich chunks with paper metadata if available
            if paper_metadata and paper_metadata.is_research_paper():
                logger.info("📊 Enriching chunks with research paper metadata...")
                chunks = self.chunker.enrich_with_paper_metadata(chunks, paper_metadata)
            
            result.chunks = chunks
            result.total_chunks = len(chunks)
            
            avg_tokens = sum(c.metadata.get('token_count', 0) for c in chunks) / len(chunks) if chunks else 0
            logger.success(
                f"✅ Created {len(chunks)} chunks (avg {avg_tokens:.0f} tokens/chunk)"
            )
            
            # ================================================================
            # STAGE 5: DATABASE STORAGE
            # ================================================================
            logger.info("💾 Stage 5/5: Storing in database...")
            
            # Prepare document record
            doc_metadata = metadata or {}
            doc_metadata.update({
                'original_path': str(file_path),
                'file_size_bytes': file_path.stat().st_size,
                'conversion_metadata': conversion_result.metadata,
                'preprocessed': True,
                'images_processed': len(image_descriptions)
            })
            
            # Add paper metadata if this is a research paper
            if paper_metadata and paper_metadata.is_research_paper():
                doc_metadata['paper_metadata'] = paper_metadata.to_dict()
                doc_metadata['is_research_paper'] = True
                logger.info("  📚 Stored research paper metadata")
            else:
                doc_metadata['is_research_paper'] = False
            
            document_record = DocumentRecord(
                document_id=document_id,
                filename=file_path.name,
                file_path=str(file_path),
                file_type=file_path.suffix.lstrip('.'),
                total_chunks=len(chunks),
                processing_date=datetime.now().isoformat(),
                metadata=doc_metadata,
                user_id=user_id
            )
            
            # Store document
            self.storage.add_document(document_record)
            logger.info("  ✅ Stored document record")
            
            # Convert chunks to ChunkRecords
            chunk_records = [
                ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.metadata.get('document_id', document_id),
                    chunk_index=chunk.metadata.get('chunk_index', idx),
                    text=chunk.text,
                    token_count=chunk.metadata.get('token_count', 0),
                    header_path=chunk.metadata.get('header_path', ''),
                   page_number=chunk.metadata.get('page_number'),
                    has_code=chunk.metadata.get('has_code', False),
                    has_tables=chunk.metadata.get('has_tables', False),
                    has_overlap=chunk.metadata.get('has_overlap', False),
                    merged=chunk.metadata.get('merged', False),
                    emergency_split=chunk.metadata.get('emergency_split', False),
                    metadata=chunk.metadata
                )
                for idx, chunk in enumerate(chunks)
            ]
            
            # Bulk insert chunks (much faster!)
            self.storage.add_chunks_bulk(chunk_records)
            logger.info(f"  ✅ Stored {len(chunks)} chunks")
            
            # Store images
            if image_descriptions:
                for img_desc in image_descriptions:
                    image_record = ImageRecord(
                        image_id=img_desc.image_id,
                        document_id=document_id,
                        image_path=img_desc.image_path,
                        description=img_desc.description,
                        ocr_text=img_desc.ocr_text,
                        vision_provider=img_desc.provider.value,
                        metadata=img_desc.metadata
                    )
                    self.storage.add_image(image_record)
                
                logger.info(f"  ✅ Stored {len(image_descriptions)} images")
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            result.processing_time_seconds = processing_time
            
            # Mark as successful
            result.success = True
            
            logger.success(f"🎉 Pipeline complete! ({processing_time:.1f}s total)")
            logger.info(f"   Document: {document_id}")
            logger.info(f"   Chunks: {len(chunks)}")
            logger.info(f"   Images: {len(image_descriptions)}")
            
            return result
        
        except Exception as e:
            # Catch any unexpected errors
            result.error = str(e)
            result.error_stage = "unknown"
            logger.error(f"❌ Pipeline failed with unexpected error: {e}")
            logger.exception(e)
            return result
    
    def process_batch(
        self,
        file_paths: List[str | Path],
        skip_errors: bool = True
    ) -> List[PipelineResult]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of paths to documents
            skip_errors: Continue processing if a document fails (default True)
        
        Returns:
            List of PipelineResults (one per document)
        """
        logger.info(f"📚 Batch processing {len(file_paths)} documents...")
        
        results = []
        
        for file_path in tqdm(file_paths, desc="Processing documents", unit="doc"):
            try:
                result = self.process_document(file_path)
                results.append(result)
                
                if not result.success and not skip_errors:
                    logger.error(f"❌ Stopping batch processing due to error in {file_path}")
                    break
            
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
                if not skip_errors:
                    raise
        
        # Summary
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes
        total_chunks = sum(r.total_chunks for r in results)
        total_images = sum(r.total_images for r in results)
        
        logger.success("✅ Batch processing complete!")
        logger.info(f"   Successes: {successes}/{len(results)}")
        logger.info(f"   Failures: {failures}")
        logger.info(f"   Total chunks: {total_chunks}")
        logger.info(f"   Total images: {total_images}")
        
        return results
    
    def _generate_document_id(self, file_path: Path) -> str:
        """
        Generate a unique document ID based on file path and name.
        
        Args:
            file_path: Path to document
        
        Returns:
            Unique document ID (format: doc_<hash>)
        """
        # Use file path + name for uniqueness
        content = f"{file_path.absolute()}_{file_path.name}"
        hash_hex = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"doc_{hash_hex}"
    
    def get_document_stats(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a processed document.
        
        Args:
            document_id: Document ID to query
        
        Returns:
            Dict with document stats or None if not found
        """
        doc = self.storage.get_document(document_id)
        if not doc:
            return None
        
        chunks = self.storage.get_chunks_by_document(document_id)
        images = self.storage.get_images_by_document(document_id)
        
        return {
            'document_id': doc.document_id,
            'filename': doc.filename,
            'file_type': doc.file_type,
            'total_chunks': len(chunks),
            'total_images': len(images),
            'avg_chunk_tokens': sum(c.token_count for c in chunks) / len(chunks) if chunks else 0,
            'chunks_with_code': sum(1 for c in chunks if c.has_code),
            'chunks_with_tables': sum(1 for c in chunks if c.has_tables),
            'processing_date': doc.processing_date,
            'metadata': doc.metadata
        }
    
    def close(self):
        """Close the storage connection."""
        if self.storage:
            self.storage.close()
            logger.info("Pipeline closed")


# Convenience function for quick processing
def process_document(
    file_path: str | Path,
    enable_images: bool = True
) -> PipelineResult:
    """
    Quick convenience function to process a single document.
    
    Args:
        file_path: Path to document
        enable_images: Whether to process images (default True)
    
    Returns:
        PipelineResult
    """
    pipeline = DocumentPipeline(enable_images=enable_images)
    result = pipeline.process_document(file_path)
    pipeline.close()
    return result
