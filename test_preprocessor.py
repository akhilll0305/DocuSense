"""
Test script for Text Preprocessor.

Demonstrates text cleaning with realistic examples.
"""

from docusense.ingestion.preprocessor import TextPreprocessor, preprocess_text
from loguru import logger

# Configure logging
logger.add("logs/preprocessor_demo.log", rotation="10 MB")


def demo_text_preprocessor():
    """
    Demonstrate TextPreprocessor with realistic examples.
    
    WHAT THIS SHOWS:
    ----------------
    1. Unicode normalization (curly quotes → straight)
    2. Whitespace cleaning (multiple spaces → single)
    3. Newline collapsing (\\n\\n\\n\\n → \\n\\n)
    4. Code block preservation
    5. Statistics tracking
    """
    print("\n" + "="*70)
    print("  Text Preprocessor Demo")
    print("="*70 + "\n")
    
    # Initialize preprocessor
    print("🔧 Initializing TextPreprocessor...")
    preprocessor = TextPreprocessor(
        normalize_unicode=True,
        remove_extra_whitespace=True,
        max_consecutive_newlines=2,
        preserve_code_blocks=True
    )
    print("   ✅ Initialized with default settings\n")
    
    # Example 1: Unicode normalization
    print("=" * 70)
    print("Example 1: Unicode Normalization")
    print("=" * 70 + "\n")
    
    messy_unicode = """The CEO said "revenue increased" and shareholders' confidence grew.
We've achieved 100% growth—that's remarkable!
Key metrics:
• Revenue: $5M
• Growth: 45%
Here's what we learned…"""
    
    print("BEFORE:")
    print("-" * 70)
    print(repr(messy_unicode))
    print("-" * 70)
    
    result1 = preprocessor.process(messy_unicode)
    
    print("\nAFTER:")
    print("-" * 70)
    print(repr(result1.cleaned_text))
    print("-" * 70)
    
    print("\n📊 Changes:")
    print(f"   Curly quotes (" ") → straight quotes (\")")
    print(f"   Apostrophes (') → straight apostrophes (')")
    print(f"   Em dashes (—) → hyphens (-)")
    print(f"   Bullets (•) → hyphens (-)")
    print(f"   Ellipsis (…) → three dots (...)")
    print(f"\n   Unicode normalized: {result1.stats.get('unicode_normalized', 0)} chars")
    
    # Example 2: Whitespace cleaning
    print("\n" + "=" * 70)
    print("Example 2: Whitespace Cleaning")
    print("=" * 70 + "\n")
    
    messy_spaces = """Revenue    increased    by    45%    in    Q4.


Our    strategy    focused    on:
-  Customer   acquisition
-  Product    development
-  Market     expansion


Key    results    were    impressive."""
    
    print("BEFORE:")
    print("-" * 70)
    print(repr(messy_spaces[:100]) + "...")
    print("-" * 70)
    
    result2 = preprocessor.process(messy_spaces)
    
    print("\nAFTER:")
    print("-" * 70)
    print(result2.cleaned_text)
    print("-" * 70)
    
    print("\n📊 Changes:")
    print(f"   Multiple spaces → single space")
    print(f"   Excessive newlines collapsed")
    print(f"   Spaces normalized: {result2.stats.get('spaces_normalized', 0)}")
    print(f"   Newlines collapsed: {result2.stats.get('newlines_collapsed', 0)}")
    
    # Example 3: Code block preservation
    print("\n" + "=" * 70)
    print("Example 3: Code Block Preservation")
    print("=" * 70 + "\n")
    
    text_with_code = """Here's    how    to    use    the    API:

```python
def  calculate_revenue(  sales,    expenses  ):
    profit   =   sales   -   expenses
    return    profit
```

The    code    above    shows    our    calculation."""
    
    print("BEFORE:")
    print("-" * 70)
    print(text_with_code)
    print("-" * 70)
    
    result3 = preprocessor.process(text_with_code)
    
    print("\nAFTER:")
    print("-" * 70)
    print(result3.cleaned_text)
    print("-" * 70)
    
    print("\n📊 Changes:")
    print(f"   Code block preserved: {result3.stats.get('code_blocks_preserved', 0)}")
    print(f"   Spaces cleaned OUTSIDE code block")
    print(f"   Spaces INSIDE code block preserved (exact formatting)")
    
    # Example 4: PDF extraction artifacts
    print("\n" + "=" * 70)
    print("Example 4: Page Artifact Removal")
    print("=" * 70 + "\n")
    
    pdf_text = """Page 1 | Company Confidential

## Introduction

Our revenue grew significantly in 2024.

Page 2 | Company Confidential

## Market Analysis

We captured 45% market share.

Page 3 | Company Confidential

## Financial Results

Revenue increased to $5M.

Page 4 | Company Confidential

## Conclusion

Strong performance across all metrics.

Page 5 | Company Confidential"""
    
    print("BEFORE:")
    print("-" * 70)
    print(pdf_text[:150] + "...")
    print("-" * 70)
    
    result4 = preprocessor.process(pdf_text)
    
    print("\nAFTER:")
    print("-" * 70)
    print(result4.cleaned_text)
    print("-" * 70)
    
    print("\n📊 Changes:")
    print(f"   Page artifacts removed: {result4.stats.get('page_artifacts_removed', 0)}")
    print(f"   Note: 'Page X | Company Confidential' appeared 5 times → removed")
    
    # Example 5: Combined realistic example
    print("\n" + "=" * 70)
    print("Example 5: Real-World Document (Combined Issues)")
    print("=" * 70 + "\n")
    
    realistic_doc = """Page 1 | Annual Report 2024


# Annual    Report    2024


## Executive    Summary

The CEO said "we've   achieved   remarkable    growth" with shareholders' confidence at an all-time high.



Key   metrics:
•  Revenue:    $5M   (45%   increase)
•  Customers:    10,000+
•  Retention:   95%



Page 2 | Annual Report 2024

## Code   Example

Here's    our    API:

```python
def   get_revenue(   year   ):
    return    database.query(   year   )
```

The    code    runs    efficiently…


Page 3 | Annual Report 2024"""
    
    print("BEFORE:")
    print("-" * 70)
    print(f"Length: {len(realistic_doc)} characters")
    print(realistic_doc[:200] + "...")
    print("-" * 70)
    
    result5 = preprocessor.process(realistic_doc)
    
    print("\nAFTER:")
    print("-" * 70)
    print(f"Length: {len(result5.cleaned_text)} characters")
    print(result5.cleaned_text)
    print("-" * 70)
    
    print("\n📊 Complete Statistics:")
    for key, value in result5.stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n📉 Size reduction: {result5.stats['reduction_percent']}%")
    print(f"   {result5.stats['original_length']} → {result5.stats['cleaned_length']} characters")
    
    # Example 6: Quick convenience function
    print("\n" + "=" * 70)
    print("Example 6: Quick Convenience Function")
    print("=" * 70 + "\n")
    
    quick_text = "Revenue    increased    by    45%    in    Q4."
    
    print(f"BEFORE: {repr(quick_text)}")
    
    # One-liner cleaning
    clean = preprocess_text(quick_text)
    
    print(f"AFTER:  {repr(clean)}")
    
    print("\n💡 Quick usage:")
    print("   >>> from docusense.ingestion.preprocessor import preprocess_text")
    print("   >>> clean = preprocess_text(messy_text)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: What Text Preprocessing Does")
    print("=" * 70 + "\n")
    
    print("✅ CLEANS:")
    print("   • Curly quotes → straight quotes")
    print("   • Multiple spaces → single space")
    print("   • Excessive newlines → max 2 consecutive")
    print("   • Unicode artifacts → ASCII equivalents")
    print("   • Page headers/footers (if repeated 5+ times)")
    print()
    print("✅ PRESERVES:")
    print("   • Markdown structure (headers, lists, tables)")
    print("   • Code blocks (exact formatting)")
    print("   • Paragraph breaks (semantic boundaries)")
    print("   • Meaningful newlines")
    print()
    print("📊 TYPICAL RESULTS:")
    print("   • 5-15% size reduction")
    print("   • Better embeddings (normalized text)")
    print("   • Fewer wasted tokens")
    print("   • Cleaner chunking")
    print()
    print("✅ Demo complete!\n")


if __name__ == "__main__":
    demo_text_preprocessor()
