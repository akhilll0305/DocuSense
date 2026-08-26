"""
Text Preprocessor - Cleans and normalizes text for optimal chunking/embedding.

This module is Step 5 of Phase 1: Knowledge Ingestion.

PURPOSE:
--------
Prepare Markdown text for chunking and embedding by:
1. Normalizing whitespace (not removing structure!)
2. Fixing unicode issues (smart quotes → regular quotes)
3. Removing noise (excessive newlines, artifacts)
4. Preserving semantic structure (headers, lists, code blocks)

WHY PREPROCESSING MATTERS:
--------------------------
Raw document text often contains:
- Multiple spaces/tabs: "Revenue    increased"
- Inconsistent newlines: "\n\n\n\n\n"
- Unicode artifacts: "don't" → "don't" (curly quotes)
- Page artifacts: "Page 5 | Company Name" (repeated headers)
- PDF extraction errors: "w o r d s p a c e d"

These reduce embedding quality and waste tokens!

PHILOSOPHY: "LIGHT TOUCH"
--------------------------
We DON'T want to over-clean because:
- LLMs are trained on natural text (including some noise)
- Markdown structure is crucial for chunking
- Code blocks need exact formatting
- Lists, tables, and headers guide semantic meaning

We DO want to:
- Normalize whitespace (multiple spaces → single)
- Fix obvious unicode issues
- Remove repeated headers/footers
- Collapse excessive newlines

WHAT WE PRESERVE:
-----------------
✅ Markdown headers (##, ###)
✅ Code blocks (```python...```)
✅ Lists and tables
✅ Paragraph breaks (semantic boundaries)
✅ Bold/italic formatting
✅ Links and references

WHAT WE CLEAN:
--------------
🧹 Multiple spaces → single space
🧹 Excessive newlines → max 2
🧹 Unicode curly quotes → regular quotes
🧹 Non-breaking spaces → regular spaces
🧹 Trailing whitespace
🧹 Page numbers/headers (if repeated)
"""

import re
import unicodedata
from typing import List
from dataclasses import dataclass

from loguru import logger


@dataclass
class PreprocessResult:
    """
    Result of text preprocessing.
    
    Attributes:
        original_text: Input text before cleaning
        cleaned_text: Output text after cleaning
        stats: Statistics about changes made
    """
    original_text: str
    cleaned_text: str
    stats: dict


class TextPreprocessor:
    """
    Lightweight text cleaner for Markdown documents.
    
    ARCHITECTURE:
    -------------
    1. Normalize Unicode (curly quotes, special chars)
    2. Clean whitespace (spaces, tabs, newlines)
    3. Remove artifacts (page numbers, repeated headers)
    4. Preserve structure (Markdown, code blocks)
    5. Collect statistics for debugging
    
    EXAMPLE USAGE:
    --------------
    >>> preprocessor = TextPreprocessor()
    >>> 
    >>> raw_text = "Revenue    increased\\n\\n\\n\\n\\nby 45%"
    >>> result = preprocessor.process(raw_text)
    >>> 
    >>> print(result.cleaned_text)
    # "Revenue increased\\n\\nby 45%"
    >>> 
    >>> print(result.stats)
    # {'spaces_normalized': 3, 'newlines_collapsed': 3, ...}
    """
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_extra_whitespace: bool = True,
        max_consecutive_newlines: int = 2,
        remove_page_artifacts: bool = True,
        preserve_code_blocks: bool = True
    ):
        """
        Initialize text preprocessor with configuration.
        
        Args:
            normalize_unicode: Fix curly quotes, special chars
            remove_extra_whitespace: Collapse multiple spaces
            max_consecutive_newlines: Max blank lines (2 = one blank line)
            remove_page_artifacts: Remove repeated headers/footers
            preserve_code_blocks: Don't clean inside ```code blocks```
        """
        self.normalize_unicode = normalize_unicode
        self.remove_extra_whitespace = remove_extra_whitespace
        self.max_consecutive_newlines = max_consecutive_newlines
        self.remove_page_artifacts = remove_page_artifacts
        self.preserve_code_blocks = preserve_code_blocks
        
        logger.debug("TextPreprocessor initialized with settings:")
        logger.debug(f"  Normalize Unicode: {normalize_unicode}")
        logger.debug(f"  Remove extra whitespace: {remove_extra_whitespace}")
        logger.debug(f"  Max consecutive newlines: {max_consecutive_newlines}")
    
    def process(self, text: str) -> PreprocessResult:
        """
        Clean and normalize text.
        
        PIPELINE:
        ---------
        1. Extract code blocks (preserve them)
        2. Normalize Unicode characters
        3. Clean whitespace (spaces, tabs, newlines)
        4. Remove page artifacts
        5. Restore code blocks (unchanged)
        6. Collect statistics
        
        Args:
            text: Raw text from document conversion
            
        Returns:
            PreprocessResult with cleaned text and statistics
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> result = preprocessor.process(messy_text)
            >>> print(f"Cleaned: {len(result.original_text)} → {len(result.cleaned_text)} chars")
            >>> print(f"Changes: {result.stats}")
        """
        if not text or not text.strip():
            return PreprocessResult(
                original_text=text,
                cleaned_text="",
                stats={"error": "Empty input"}
            )
        
        original_length = len(text)
        cleaned = text
        stats = {}
        
        # Step 1: Extract code blocks to preserve them
        code_blocks = []
        if self.preserve_code_blocks:
            cleaned, code_blocks = self._extract_code_blocks(cleaned)
            stats['code_blocks_preserved'] = len(code_blocks)
        
        # Step 2: Normalize Unicode
        if self.normalize_unicode:
            cleaned, unicode_changes = self._normalize_unicode(cleaned)
            stats['unicode_normalized'] = unicode_changes
        
        # Step 3: Clean whitespace
        if self.remove_extra_whitespace:
            cleaned, space_changes = self._clean_whitespace(cleaned)
            stats['spaces_normalized'] = space_changes
        
        # Step 4: Collapse excessive newlines
        cleaned, newline_changes = self._collapse_newlines(
            cleaned, 
            max_consecutive=self.max_consecutive_newlines
        )
        stats['newlines_collapsed'] = newline_changes
        
        # Step 5: Remove page artifacts (optional)
        if self.remove_page_artifacts:
            cleaned, artifacts_removed = self._remove_page_artifacts(cleaned)
            stats['page_artifacts_removed'] = artifacts_removed
        
        # Step 6: Restore code blocks
        if self.preserve_code_blocks and code_blocks:
            cleaned = self._restore_code_blocks(cleaned, code_blocks)
        
        # Step 7: Final trim
        cleaned = cleaned.strip()
        
        # Collect final stats
        stats['original_length'] = original_length
        stats['cleaned_length'] = len(cleaned)
        stats['reduction_percent'] = round(
            (1 - len(cleaned) / original_length) * 100, 1
        ) if original_length > 0 else 0
        
        logger.info(f"Text preprocessed: {original_length} → {len(cleaned)} chars (-{stats['reduction_percent']}%)")
        
        return PreprocessResult(
            original_text=text,
            cleaned_text=cleaned,
            stats=stats
        )
    
    def _extract_code_blocks(self, text: str) -> tuple[str, List[str]]:
        """
        Extract code blocks to preserve exact formatting.
        
        Code blocks (```...```) need exact whitespace for:
        - Python indentation
        - JSON structure
        - Command examples
        
        We replace them with placeholders, clean the rest,
        then restore them unchanged.
        
        Args:
            text: Input text with code blocks
            
        Returns:
            Tuple of (text_with_placeholders, extracted_code_blocks)
        """
        code_blocks = []
        
        # Match triple-backtick code blocks
        pattern = r'```[\s\S]*?```'
        
        def replace_code_block(match):
            code_blocks.append(match.group(0))
            return f"<<<CODE_BLOCK_{len(code_blocks) - 1}>>>"
        
        text_with_placeholders = re.sub(pattern, replace_code_block, text)
        
        return text_with_placeholders, code_blocks
    
    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """
        Restore code blocks to their original form.
        
        Args:
            text: Text with placeholders
            code_blocks: Original code block content
            
        Returns:
            Text with code blocks restored
        """
        for i, code_block in enumerate(code_blocks):
            placeholder = f"<<<CODE_BLOCK_{i}>>>"
            text = text.replace(placeholder, code_block)
        
        return text
    
    def _normalize_unicode(self, text: str) -> tuple[str, int]:
        """
        Normalize Unicode characters to ASCII equivalents where possible.
        
        COMMON ISSUES:
        --------------
        - Curly quotes: " " → " "
        - Apostrophes: ' → '
        - Em/en dashes: — – → -
        - Non-breaking spaces: \\xa0 → space
        - Ellipsis: … → ...
        
        WHY: LLMs tokenize ASCII more efficiently, and
        embeddings work better with consistent characters.
        
        Args:
            text: Input text with unicode chars
            
        Returns:
            Tuple of (normalized_text, changes_made)
        """
        changes = 0
        
        # Specific replacements for common issues
        replacements = {
            # Curly quotes -> straight quotes. Written as escapes: the literal
            # characters were lost to an encoding round-trip, leaving four
            # entries that mapped ASCII quotes to themselves (and duplicated keys).
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
            
            # Dashes
            '—': '-',  # Em dash
            '–': '-',  # En dash
            
            # Spaces
            '\xa0': ' ',  # Non-breaking space
            '\u200b': '',  # Zero-width space
            
            # Ellipsis
            '…': '...',
            
            # Bullets
            '•': '-',
            '·': '-',
        }
        
        for old_char, new_char in replacements.items():
            if old_char in text:
                # Count before replacing: afterwards old_char is gone and the
                # tally was always zero.
                changes += text.count(old_char)
                text = text.replace(old_char, new_char)
        
        # Normalize remaining unicode (NFD → NFC)
        # This handles accented characters: é → e
        # (We keep accents, just normalize them)
        text = unicodedata.normalize('NFC', text)
        
        return text, changes
    
    def _clean_whitespace(self, text: str) -> tuple[str, int]:
        """
        Normalize whitespace (spaces and tabs).
        
        CLEANING:
        ---------
        - Multiple spaces → single space
        - Tabs → spaces
        - Trailing whitespace on lines
        - Leading whitespace (preserve indentation relatively)
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (cleaned_text, changes_made)
        """
        original = text
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Multiple spaces → single space (but not at line start)
        # This preserves some indentation
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            
            # Collapse multiple spaces (but preserve leading indentation somewhat)
            # Split into leading spaces and content
            leading_spaces = len(line) - len(line.lstrip())
            content = line.lstrip()
            
            # Collapse multiple spaces in content
            content = re.sub(r' +', ' ', content)
            
            # Preserve up to 4 leading spaces (for lists, indentation)
            leading = ' ' * min(leading_spaces, 4)
            
            cleaned_lines.append(leading + content)
        
        text = '\n'.join(cleaned_lines)
        
        # Count changes
        changes = len(original) - len(text)
        
        return text, max(0, changes)
    
    def _collapse_newlines(self, text: str, max_consecutive: int = 2) -> tuple[str, int]:
        """
        Collapse excessive newlines.
        
        RATIONALE:
        ----------
        Excessive blank lines waste tokens and don't add meaning:
        "\\n\\n\\n\\n\\n" → "\\n\\n" (one blank line is enough)
        
        But we preserve SOME newlines for paragraph breaks
        (semantic boundaries for chunking).
        
        Args:
            text: Input text
            max_consecutive: Max consecutive newlines (2 = one blank line)
            
        Returns:
            Tuple of (cleaned_text, newlines_removed)
        """
        original_newlines = text.count('\n')
        
        # Replace 3+ newlines with max_consecutive newlines
        pattern = r'\n{' + str(max_consecutive + 1) + r',}'
        replacement = '\n' * max_consecutive
        
        text = re.sub(pattern, replacement, text)
        
        final_newlines = text.count('\n')
        changes = original_newlines - final_newlines
        
        return text, max(0, changes)
    
    def _remove_page_artifacts(self, text: str) -> tuple[str, int]:
        """
        Remove repeated page headers/footers from PDF extraction.
        
        COMMON ARTIFACTS:
        -----------------
        - "Page 5 | Company Name" (repeated every page)
        - "Confidential - Do Not Distribute" (footer)
        - "Chapter 2 - Introduction" (running header)
        
        DETECTION:
        ----------
        If same line appears 3+ times, it's likely an artifact.
        
        CAUTION:
        --------
        Be conservative - don't remove actual content!
        Only remove if appears 5+ times.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (cleaned_text, artifacts_removed)
        """
        lines = text.split('\n')
        
        # Count line frequencies
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 5:  # Ignore very short lines
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        
        # Find likely artifacts (repeated 5+ times)
        artifacts = {
            line for line, count in line_counts.items() 
            if count >= 5 and len(line) < 100  # Short repeated lines
        }
        
        if not artifacts:
            return text, 0
        
        logger.debug(f"Found {len(artifacts)} likely page artifacts")
        
        # Remove artifacts
        cleaned_lines = []
        removed = 0
        
        for line in lines:
            if line.strip() in artifacts:
                removed += 1
            else:
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        return text, removed
    
    def batch_process(self, texts: List[str]) -> List[PreprocessResult]:
        """
        Process multiple texts efficiently.
        
        Args:
            texts: List of text strings to clean
            
        Returns:
            List of PreprocessResult objects
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.debug(f"Processing text {i + 1}/{len(texts)}")
            result = self.process(text)
            results.append(result)
        
        # Log batch statistics
        total_original = sum(r.stats['original_length'] for r in results)
        total_cleaned = sum(r.stats['cleaned_length'] for r in results)
        avg_reduction = (1 - total_cleaned / total_original) * 100 if total_original > 0 else 0
        
        logger.info(f"Batch processed {len(texts)} texts: {total_original} → {total_cleaned} chars (-{avg_reduction:.1f}%)")
        
        return results


# Convenience function
def preprocess_text(text: str, **kwargs) -> str:
    """
    Quick preprocessing function for single texts.
    
    Args:
        text: Text to clean
        **kwargs: Options for TextPreprocessor
        
    Returns:
        Cleaned text string
        
    Example:
        >>> clean = preprocess_text(raw_text)
    """
    preprocessor = TextPreprocessor(**kwargs)
    result = preprocessor.process(text)
    return result.cleaned_text
