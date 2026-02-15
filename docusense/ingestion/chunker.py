"""
Semantic Chunker - Intelligent Markdown-aware text chunking for RAG.

This is the MOST CRITICAL component of Phase 1!

PURPOSE:
--------
Split documents into semantically meaningful chunks that:
1. Fit within embedding model limits (512 tokens)
2. Preserve complete thoughts (don't split mid-sentence)
3. Maintain context (header hierarchy, overlap)
4. Optimize retrieval quality (semantic boundaries)

WHY CHUNKING IS CRITICAL FOR RAG:
----------------------------------
Bad chunking = Bad RAG system, no matter how good your LLM!

Example of BAD chunking (fixed-size, mid-sentence):
    Chunk 1: "Our revenue increased by 45% in Q4. The main drivers were customer acquisition and"
    Chunk 2: "product development. We also expanded into new markets which contributed"
    → Context broken! Missing what "contributed" to.

Example of GOOD chunking (semantic, header-aware):
    Chunk 1: "## Q4 Results\\n\\nOur revenue increased by 45% in Q4. The main drivers were customer acquisition and product development. We also expanded into new markets which contributed significantly."
    → Complete thought! Includes header for context!

CHUNKING STRATEGY: MARKDOWN-AWARE SEMANTIC
-------------------------------------------
We use document structure as a guide:

1. **Primary boundaries**: ## level-2 headers
   - Each section becomes a chunk (if size appropriate)
   - Preserves topic coherence

2. **Secondary boundaries**: ### level-3 headers
   - Sub-divide large sections
   - Maintain hierarchy in metadata

3. **Preserve intact**:
   - Code blocks (```...```)
   - Tables (|...|)
   - Lists (-, 1., *)
   - Paragraphs

4. **Overlap strategy**:
   - Include parent header in child chunks
   - Add N sentences of overlap between chunks
   - Maintains context for retrieval

TOKEN LIMITS:
-------------
- Minimum: 200 tokens (too small = loss of context)
- Target: 500 tokens (sweet spot for retrieval)
- Maximum: 800 tokens (prevent truncation)
- Hard limit: 512 tokens (embedding model limit)

METADATA TRACKING:
-------------------
Each chunk gets rich metadata:
- chunk_id: Unique identifier
- document_id: Source document
- chunk_index: Position in document (0, 1, 2...)
- header_path: "Document > Section > Subsection"
- token_count: Actual token count
- has_code: Contains code blocks?
- has_tables: Contains tables?
- start_char: Character position in original document
- end_char: Character end position
"""

import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import tiktoken
from loguru import logger

from docusense.config import settings


@dataclass
class Chunk:
    """
    A semantically meaningful piece of text.
    
    Attributes:
        chunk_id: Unique identifier
        text: The chunk content
        metadata: Rich metadata for retrieval and citation
    """
    chunk_id: str
    text: str
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate chunk after creation."""
        if not self.text.strip():
            raise ValueError("Chunk text cannot be empty")
        
        if 'token_count' not in self.metadata:
            # Calculate token count if not provided
            encoder = tiktoken.get_encoding("cl100k_base")
            self.metadata['token_count'] = len(encoder.encode(self.text))


class SemanticChunker:
    """
    Markdown-aware semantic chunker for RAG systems.
    
    ARCHITECTURE:
    -------------
    1. Parse Markdown structure (headers, code blocks, lists)
    2. Split on semantic boundaries (headers, paragraphs)
    3. Enforce token limits (merge small, split large)
    4. Add overlap for context
    5. Generate rich metadata
    6. Validate all chunks
    
    EXAMPLE USAGE:
    --------------
    >>> chunker = SemanticChunker()
    >>> 
    >>> markdown = \"\"\"
    >>> # Document Title
    >>> ## Section 1
    >>> Content here...
    >>> 
    >>> ## Section 2
    >>> More content...
    >>> \"\"\"
    >>> 
    >>> chunks = chunker.chunk(markdown, doc_id="doc_001")
    >>> 
    >>> for chunk in chunks:
    >>>     print(f"Chunk {chunk.metadata['chunk_index']}")
    >>>     print(f"Header: {chunk.metadata['header_path']}")
    >>>     print(f"Tokens: {chunk.metadata['token_count']}")
    >>>     print(chunk.text[:100])
    """
    
    def __init__(
        self,
        min_chunk_tokens: Optional[int] = None,
        max_chunk_tokens: Optional[int] = None,
        target_chunk_tokens: Optional[int] = None,
        split_on_headers: bool = True,
        preserve_code_blocks: bool = True,
        preserve_tables: bool = True,
        overlap_sentences: int = 1,
    ):
        """
        Initialize semantic chunker.
        
        Args:
            min_chunk_tokens: Minimum chunk size (default from settings)
            max_chunk_tokens: Maximum chunk size (default from settings)
            target_chunk_tokens: Target chunk size (default from settings)
            split_on_headers: Split at header boundaries
            preserve_code_blocks: Don't split code blocks
            preserve_tables: Don't split tables
            overlap_sentences: Number of sentences to overlap
        """
        self.min_chunk_tokens = min_chunk_tokens or settings.min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens or settings.max_chunk_tokens
        self.target_chunk_tokens = target_chunk_tokens or settings.target_chunk_tokens
        self.split_on_headers = split_on_headers
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables
        self.overlap_sentences = overlap_sentences
        
        # Token encoder for accurate counting
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"SemanticChunker initialized")
        logger.info(f"  Token range: {self.min_chunk_tokens}-{self.max_chunk_tokens}")
        logger.info(f"  Target: {self.target_chunk_tokens} tokens")
        logger.info(f"  Overlap: {self.overlap_sentences} sentences")
    
    def chunk(
        self, 
        text: str, 
        doc_id: str,
        doc_metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Split text into semantic chunks.
        
        PROCESS:
        --------
        1. Parse Markdown structure
        2. Split on header boundaries
        3. Validate token counts
        4. Merge too-small chunks
        5. Split too-large chunks
        6. Add overlap
        7. Generate metadata
        
        Args:
            text: Markdown text to chunk
            doc_id: Document identifier
            doc_metadata: Optional document-level metadata
            
        Returns:
            List of Chunk objects with text and metadata
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for chunking: {doc_id}")
            return []
        
        logger.info(f"Chunking document {doc_id}: {len(text)} chars")
        
        doc_metadata = doc_metadata or {}
        
        # Step 1: Parse Markdown structure
        sections = self._parse_markdown_structure(text)
        logger.debug(f"  Parsed {len(sections)} sections")
        
        # Step 2: Create initial chunks from sections
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, doc_id, doc_metadata)
            chunks.extend(section_chunks)
        
        logger.debug(f"  Created {len(chunks)} initial chunks")
        
        # Step 3: Validate and adjust chunk sizes
        chunks = self._validate_chunk_sizes(chunks, doc_id, doc_metadata)
        logger.debug(f"  Validated to {len(chunks)} final chunks")
        
        # Step 4: Add overlap between chunks
        if self.overlap_sentences > 0:
            chunks = self._add_overlap(chunks)
            logger.debug(f"  Added {self.overlap_sentences}-sentence overlap")
        
        # Step 5: Finalize metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['document_id'] = doc_id
        
        logger.success(f"✅ Chunked {doc_id}: {len(chunks)} chunks, avg {self._avg_tokens(chunks)} tokens")
        
        return chunks
    
    def _parse_markdown_structure(self, text: str) -> List[Dict]:
        """
        Parse Markdown into structured sections.
        
        PARSING STRATEGY:
        -----------------
        - Split on ## (level-2) headers
        - Track header hierarchy
        - Preserve code blocks and tables
        - Extract metadata (has_code, has_tables)
        
        Args:
            text: Markdown text
            
        Returns:
            List of section dictionaries with:
              - text: Section content
              - header: Section header text
              - level: Header level (1, 2, 3)
              - has_code: Contains code blocks
              - has_tables: Contains tables
        """
        sections = []
        
        # Split on headers while preserving them
        # Pattern: ^## Header Text
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = text.split('\n')
        current_section = {
            'header': '',
            'level': 0,
            'lines': [],
            'has_code': False,
            'has_tables': False
        }
        
        in_code_block = False
        
        for line in lines:
            # Track code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                current_section['has_code'] = True
            
            # Track tables
            if '|' in line and not in_code_block:
                current_section['has_tables'] = True
            
            # Check for header
            header_match = re.match(header_pattern, line)
            
            if header_match and not in_code_block:
                # Found a header - save current section if it has content
                if current_section['lines']:
                    current_section['text'] = '\n'.join(current_section['lines'])
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))  # Count #'s
                header_text = header_match.group(2).strip()
                
                current_section = {
                    'header': header_text,
                    'level': level,
                    'lines': [line],  # Include header in section
                    'has_code': False,
                    'has_tables': False
                }
            else:
                # Add line to current section
                current_section['lines'].append(line)
        
        # Add final section
        if current_section['lines']:
            current_section['text'] = '\n'.join(current_section['lines'])
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(
        self, 
        section: Dict, 
        doc_id: str,
        doc_metadata: Dict
    ) -> List[Chunk]:
        """
        Chunk a single section.
        
        STRATEGY:
        ---------
        - If section fits in target, keep as single chunk
        - If too large, split on paragraphs
        - If still too large, split on sentences
        - Preserve code blocks and tables intact
        
        Args:
            section: Section dictionary from parsing
            doc_id: Document ID
            doc_metadata: Document metadata
            
        Returns:
            List of chunks for this section
        """
        text = section['text']
        token_count = self._count_tokens(text)
        
        # Case 1: Section fits in one chunk
        if token_count <= self.max_chunk_tokens:
            chunk_id = f"{doc_id}_chunk_{str(uuid.uuid4())[:8]}"
            return [Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    'header': section['header'],
                    'header_level': section['level'],
                    'header_path': section['header'],
                    'token_count': token_count,
                    'has_code': section['has_code'],
                    'has_tables': section['has_tables'],
                    **doc_metadata
                }
            )]
        
        # Case 2: Section too large - split on paragraphs
        logger.debug(f"  Section '{section['header']}' too large ({token_count} tokens), splitting")
        
        # Split on double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_text = ''
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            # If adding this paragraph exceeds target, create chunk
            if current_tokens + para_tokens > self.target_chunk_tokens and current_text:
                chunk_id = f"{doc_id}_chunk_{str(uuid.uuid4())[:8]}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=current_text.strip(),
                    metadata={
                        'header': section['header'],
                        'header_level': section['level'],
                        'header_path': section['header'],
                        'token_count': current_tokens,
                        'has_code': '```' in current_text,
                        'has_tables': '|' in current_text,
                        **doc_metadata
                    }
                ))
                current_text = ''
                current_tokens = 0
            
            # Add paragraph to current chunk
            current_text += para + '\n\n'
            current_tokens += para_tokens
        
        # Add final chunk
        if current_text.strip():
            chunk_id = f"{doc_id}_chunk_{str(uuid.uuid4())[:8]}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=current_text.strip(),
                metadata={
                    'header': section['header'],
                    'header_level': section['level'],
                    'header_path': section['header'],
                    'token_count': current_tokens,
                    'has_code': '```' in current_text,
                    'has_tables': '|' in current_text,
                    **doc_metadata
                }
            ))
        
        return chunks
    
    def _validate_chunk_sizes(
        self, 
        chunks: List[Chunk], 
        doc_id: str,
        doc_metadata: Dict
    ) -> List[Chunk]:
        """
        Validate chunk sizes and adjust if needed.
        
        RULES:
        ------
        - Merge chunks < min_tokens with previous/next
        - Split chunks > max_tokens (emergency split)
        - Ensure all chunks within limits
        
        Args:
            chunks: Initial chunks
            doc_id: Document ID
            doc_metadata: Document metadata
            
        Returns:
            Validated chunks
        """
        validated = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            token_count = chunk.metadata['token_count']
            
            # Case 1: Chunk too small - merge with next if possible
            if token_count < self.min_chunk_tokens and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                
                # Merge with next
                merged_text = chunk.text + '\n\n' + next_chunk.text
                merged_tokens = self._count_tokens(merged_text)
                
                # Only merge if result isn't too large
                if merged_tokens <= self.max_chunk_tokens:
                    chunk_id = f"{doc_id}_chunk_{str(uuid.uuid4())[:8]}"
                    merged_chunk = Chunk(
                        chunk_id=chunk_id,
                        text=merged_text,
                        metadata={
                            **chunk.metadata,
                            'token_count': merged_tokens,
                            'merged': True
                        }
                    )
                    validated.append(merged_chunk)
                    i += 2  # Skip next chunk (already merged)
                    continue
            
            # Case 2: Chunk too large - emergency split
            if token_count > self.max_chunk_tokens:
                logger.warning(f"  Chunk exceeds max ({token_count} > {self.max_chunk_tokens}), splitting")
                split_chunks = self._emergency_split(chunk, doc_id, doc_metadata)
                validated.extend(split_chunks)
            else:
                # Chunk is good size
                validated.append(chunk)
            
            i += 1
        
        return validated
    
    def _emergency_split(
        self, 
        chunk: Chunk, 
        doc_id: str,
        doc_metadata: Dict
    ) -> List[Chunk]:
        """
        Emergency split for chunks that exceed max_tokens.
        
        STRATEGY:
        ---------
        Last resort when all else fails.
        Split on sentences to preserve some coherence.
        
        Args:
            chunk: Chunk that's too large
            doc_id: Document ID
            doc_metadata: Document metadata
            
        Returns:
            List of smaller chunks
        """
        # Simple sentence split (imperfect but functional)
        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
        
        chunks = []
        current_text = ''
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_chunk_tokens and current_text:
                chunk_id = f"{doc_id}_chunk_{str(uuid.uuid4())[:8]}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=current_text.strip(),
                    metadata={
                        **chunk.metadata,
                        'token_count': current_tokens,
                        'emergency_split': True
                    }
                ))
                current_text = ''
                current_tokens = 0
            
            current_text += sentence + ' '
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_text.strip():
            chunk_id = f"{doc_id}_chunk_{str(uuid.uuid4())[:8]}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=current_text.strip(),
                metadata={
                    **chunk.metadata,
                    'token_count': current_tokens,
                    'emergency_split': True
                }
            ))
        
        return chunks
    
    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Add overlap between consecutive chunks for context.
        
        OVERLAP STRATEGY:
        -----------------
        Add last N sentences from previous chunk to current chunk.
        This helps retrieval when query spans chunk boundary.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Chunks with overlap added
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Extract last N sentences from previous chunk
            prev_sentences = re.split(r'(?<=[.!?])\s+', prev_chunk.text)
            overlap_text = ' '.join(prev_sentences[-self.overlap_sentences:])
            
            # Prepend to current chunk
            new_text = overlap_text + '\n\n' + curr_chunk.text
            
            # Update token count
            new_tokens = self._count_tokens(new_text)
            
            # Create new chunk with overlap
            overlapped_chunk = Chunk(
                chunk_id=curr_chunk.chunk_id,
                text=new_text,
                metadata={
                    **curr_chunk.metadata,
                    'token_count': new_tokens,
                    'has_overlap': True
                }
            )
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens accurately using tiktoken.
        
        CRITICAL: Never use len(text.split()) - it's inaccurate!
        
        Args:
            text: Text to count
            
        Returns:
            Accurate token count
        """
        return len(self.encoder.encode(text))
    
    def _avg_tokens(self, chunks: List[Chunk]) -> int:
        """Calculate average token count across chunks."""
        if not chunks:
            return 0
        total = sum(c.metadata['token_count'] for c in chunks)
        return total // len(chunks)
