"""
Test script for SemanticChunker - demonstrates intelligent Markdown chunking.

This shows why semantic chunking is CRITICAL for RAG quality!
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from docusense.ingestion.chunker import SemanticChunker
from loguru import logger

# Configure logger for demo
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


def separator(title: str):
    """Print a nice separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_chunk_info(chunks):
    """Print detailed chunk information."""
    print(f"\n📊 CHUNKING RESULTS:")
    print(f"  Total chunks: {len(chunks)}")
    
    if chunks:
        tokens = [c.metadata['token_count'] for c in chunks]
        print(f"  Token range: {min(tokens)}-{max(tokens)} tokens")
        print(f"  Average: {sum(tokens) // len(tokens)} tokens")
    
    print()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}/{len(chunks)}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Header: {chunk.metadata.get('header', 'N/A')}")
        print(f"  Tokens: {chunk.metadata['token_count']}")
        print(f"  Has code: {chunk.metadata.get('has_code', False)}")
        print(f"  Has tables: {chunk.metadata.get('has_tables', False)}")
        print(f"  Has overlap: {chunk.metadata.get('has_overlap', False)}")
        
        # Show first 200 characters
        preview = chunk.text.replace('\n', ' ')[:200]
        print(f"  Preview: {preview}...")
        print()


# ============================================================================
# EXAMPLE 1: Simple header-based splitting
# ============================================================================
separator("EXAMPLE 1: Header-Based Splitting")

print("📝 This demonstrates splitting on ## headers (semantic boundaries)\n")

markdown_simple = """# Research Report

## Introduction

This is the introduction section. It provides background and context for the research.
The key question we're addressing is: how does semantic chunking improve RAG systems?

## Methodology

We conducted experiments with three chunking strategies:
1. Fixed-size chunking (500 tokens, no overlap)
2. Sentence-based chunking (split on periods)
3. Semantic chunking (our approach)

Each strategy was evaluated on retrieval accuracy and answer quality.

## Results

The results show clear advantages for semantic chunking:
- 35% improvement in retrieval accuracy
- 28% improvement in answer quality
- Better preservation of context

## Conclusion

Semantic chunking is superior because it preserves complete thoughts and maintains context through header hierarchy.
"""

chunker = SemanticChunker(
    target_chunk_tokens=200,  # Lower target to force splitting
    overlap_sentences=0  # No overlap for clarity
)

chunks = chunker.chunk(markdown_simple, doc_id="example1")

print_chunk_info(chunks)

print("✅ KEY INSIGHT:")
print("  Each section becomes a separate chunk!")
print("  This preserves semantic boundaries and makes retrieval more precise.")


# ============================================================================
# EXAMPLE 2: Large section splitting
# ============================================================================
separator("EXAMPLE 2: Large Section Splitting")

print("📝 When a section is too large, it's split on paragraphs\n")

markdown_large = """## Large Section

This is the first paragraph. It contains some important information about our topic.
We want to make sure this paragraph stays intact and doesn't get split mid-sentence.

This is the second paragraph. It builds on the first paragraph and adds more context.
The connection between paragraphs matters, so we preserve complete thoughts.

This is the third paragraph. It introduces new concepts that are related but distinct.
Notice how each paragraph is a natural semantic unit.

This is the fourth paragraph. It continues the discussion with additional evidence.
By splitting here, we maintain coherence while staying within token limits.

This is the fifth paragraph. It concludes the section with a summary.
Each chunk will have complete paragraphs, never broken mid-thought.
"""

chunker_large = SemanticChunker(
    target_chunk_tokens=100,  # Very low to force paragraph splitting
    overlap_sentences=0
)

chunks_large = chunker_large.chunk(markdown_large, doc_id="example2")

print_chunk_info(chunks_large)

print("✅ KEY INSIGHT:")
print("  Paragraphs are kept intact (never split mid-paragraph)!")
print("  This preserves complete thoughts even when sections are large.")


# ============================================================================
# EXAMPLE 3: Code block preservation
# ============================================================================
separator("EXAMPLE 3: Code Block Preservation")

print("📝 Code blocks stay intact - NEVER split a code block!\n")

markdown_code = """## API Documentation

Here's how to use our API:

```python
def process_document(file_path: str) -> List[Chunk]:
    # Load document
    with open(file_path, 'r') as f:
        text = f.read()
    
    # Convert to markdown
    converter = DocumentConverter()
    result = converter.convert(file_path)
    
    # Chunk the markdown
    chunker = SemanticChunker()
    chunks = chunker.chunk(result.markdown, doc_id="doc_001")
    
    return chunks
```

This code demonstrates the complete document processing pipeline.
Notice how it integrates all the components we've built.

## Error Handling

Always handle errors gracefully:

```python
try:
    chunks = process_document("my_file.pdf")
except Exception as e:
    logger.error(f"Failed to process: {e}")
    chunks = []
```
"""

chunker_code = SemanticChunker(
    target_chunk_tokens=150,
    preserve_code_blocks=True
)

chunks_code = chunker_code.chunk(markdown_code, doc_id="example3")

print_chunk_info(chunks_code)

print("✅ KEY INSIGHT:")
print("  'has_code' metadata is True! Code blocks detected and preserved!")
print("  This ensures code snippets stay complete and don't break mid-function.")


# ============================================================================
# EXAMPLE 4: Table preservation
# ============================================================================
separator("EXAMPLE 4: Table Preservation")

print("📝 Tables stay intact - NEVER split a table across chunks!\n")

markdown_table = """## Performance Comparison

Here's how different chunking strategies compare:

| Strategy | Accuracy | Speed | Memory |
|----------|----------|-------|--------|
| Fixed-size | 65% | Fast | Low |
| Sentence-based | 72% | Medium | Medium |
| Semantic | 87% | Medium | Medium |

The table clearly shows semantic chunking achieves the best accuracy.

## Model Comparison

Different models have different token limits:

| Model | Context Window | Embedding Dim |
|-------|----------------|---------------|
| all-MiniLM-L6-v2 | 512 | 384 |
| text-embedding-ada-002 | 8191 | 1536 |
| Llama 3.2 | 128000 | 4096 |
"""

chunker_table = SemanticChunker(target_chunk_tokens=150)

chunks_table = chunker_table.chunk(markdown_table, doc_id="example4")

print_chunk_info(chunks_table)

print("✅ KEY INSIGHT:")
print("  'has_tables' metadata is True! Tables detected and preserved!")
print("  This ensures tabular data stays complete for accurate QA.")


# ============================================================================
# EXAMPLE 5: Small section merging
# ============================================================================
separator("EXAMPLE 5: Small Section Merging")

print("📝 Tiny sections get merged to avoid context loss\n")

markdown_small = """## Title

Just a short intro.

## First Point

One sentence here.

## Second Point

Another sentence there.

## Third Point

And a final sentence.

## Conclusion

Small sections like these would create tiny chunks with no context.
By merging them, we get better retrieval quality.
"""

chunker_merge = SemanticChunker(
    min_chunk_tokens=50,  # Merge chunks smaller than this
    target_chunk_tokens=150
)

chunks_merge = chunker_merge.chunk(markdown_small, doc_id="example5")

print_chunk_info(chunks_merge)

print("✅ KEY INSIGHT:")
print("  'merged' metadata indicates combined sections!")
print("  This prevents tiny chunks that lack context.")


# ============================================================================
# EXAMPLE 6: Overlap for context
# ============================================================================
separator("EXAMPLE 6: Overlap Between Chunks")

print("📝 Overlap helps when queries span chunk boundaries\n")

markdown_overlap = """## Background

The history of RAG systems begins with information retrieval.
Early systems used keyword matching and TF-IDF scoring.

## Modern Approaches

Modern RAG systems use dense embeddings for similarity search.
This allows semantic matching instead of just keyword matching.
The connection to retrieval-augmented generation came later.

## Future Directions

We expect RAG systems to become more sophisticated.
Integration with reasoning and planning will improve quality.
"""

chunker_overlap = SemanticChunker(
    target_chunk_tokens=100,
    overlap_sentences=2  # Last 2 sentences overlap into next chunk
)

chunks_overlap = chunker_overlap.chunk(markdown_overlap, doc_id="example6")

print_chunk_info(chunks_overlap)

print("✅ KEY INSIGHT:")
print("  'has_overlap' metadata is True for chunks 2+!")
print("  Each chunk includes the last 2 sentences from the previous chunk.")
print("  This maintains context when queries reference adjacent sections.")


# ============================================================================
# EXAMPLE 7: Real-world complex document
# ============================================================================
separator("EXAMPLE 7: Real-World Complex Document")

print("📝 Realistic document with headers, code, tables, and lists\n")

markdown_complex = """# Product Documentation

## Overview

Welcome to our document processing system. This guide will walk you through
the complete workflow from ingestion to retrieval.

### Key Features

- **Universal conversion**: Handles PDF, DOCX, PPTX, XLSX
- **Image understanding**: Extracts meaning from charts and diagrams
- **Semantic chunking**: Intelligent text splitting
- **Hybrid retrieval**: Combines vector and keyword search

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up environment:

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

Here's a minimal example:

```python
from docusense.ingestion import DocumentConverter, SemanticChunker

# Convert document
converter = DocumentConverter()
result = converter.convert("my_document.pdf")

# Chunk the text
chunker = SemanticChunker()
chunks = chunker.chunk(result.markdown, doc_id="doc_001")

print(f"Created {len(chunks)} chunks")
```

## Configuration

Key settings in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| MIN_CHUNK_TOKENS | 200 | Minimum chunk size |
| MAX_CHUNK_TOKENS | 800 | Maximum chunk size |
| TARGET_CHUNK_TOKENS | 500 | Target chunk size |
| GEMINI_API_KEY | - | API key for vision |

Adjust these based on your embedding model's context window.

## API Reference

### DocumentConverter

Converts documents to Markdown:

```python
class DocumentConverter:
    def convert(self, file_path: str) -> ConversionResult:
        \"\"\"Convert document to Markdown.\"\"\"
        pass
```

### SemanticChunker

Splits text into semantic chunks:

```python
class SemanticChunker:
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        \"\"\"Create semantic chunks.\"\"\"
        pass
```

## Best Practices

1. **Always preprocess**: Clean text before chunking
2. **Use semantic boundaries**: Split on headers and paragraphs
3. **Add overlap**: Helps with query spanning chunks
4. **Validate sizes**: Ensure chunks fit embedding model limits
5. **Track metadata**: Essential for citation and debugging

## Troubleshooting

### Chunks too small

Increase `MIN_CHUNK_TOKENS` or reduce header splitting.

### Chunks too large

Decrease `MAX_CHUNK_TOKENS` or enable paragraph splitting.

### Poor retrieval

Try increasing `overlap_sentences` or adjusting `target_chunk_tokens`.

## Conclusion

Semantic chunking is the foundation of a good RAG system.
Take time to tune parameters for your specific use case.
"""

chunker_complex = SemanticChunker(
    min_chunk_tokens=200,
    max_chunk_tokens=800,
    target_chunk_tokens=500,
    overlap_sentences=1
)

chunks_complex = chunker_complex.chunk(markdown_complex, doc_id="example7_docs")

print_chunk_info(chunks_complex)

print("✅ KEY INSIGHTS:")
print("  - Headers at different levels (##, ###) detected")
print("  - Code blocks preserved in multiple chunks")
print("  - Table preserved intact")
print("  - Lists maintained within their sections")
print("  - Chunks sized appropriately for embeddings")
print("  - Overlap maintains context between chunks")
print()
print("This is what a real RAG document looks like after chunking! 🎯")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
separator("SUMMARY: Why Semantic Chunking Matters")

print("""
🎓 KEY LEARNINGS:

1. **Header-based splitting** preserves document structure
   → Each section is a semantic unit

2. **Paragraph-based splitting** for large sections maintains coherence
   → Never split mid-paragraph or mid-sentence

3. **Code blocks stay intact** - essential for technical docs
   → Code snippets must be complete to be useful

4. **Tables stay intact** - critical for structured data
   → Split tables break QA systems

5. **Small sections get merged** - prevents context loss
   → Tiny chunks have insufficient information

6. **Overlap maintains context** - helps span chunk boundaries
   → Last N sentences from previous chunk included

7. **Rich metadata** enables smart retrieval
   → Headers, token counts, content flags

🔥 THE BOTTOM LINE:

Bad chunking = Bad RAG, no matter how good your LLM!

Semantic chunking creates:
✅ Complete thoughts
✅ Preserved context
✅ Optimal token usage
✅ Better retrieval
✅ Higher quality answers

This is THE MOST IMPORTANT STEP in building a RAG system! 🎯
""")
