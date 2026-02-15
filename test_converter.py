"""
Test script for Document Converter.

This demonstrates how the DocumentConverter works with different file types.
"""

from pathlib import Path
from docusense.ingestion import DocumentConverter
from loguru import logger

# Configure logger for demo
logger.add("logs/converter_demo.log", rotation="10 MB")


def demo_converter():
    """
    Demonstrate DocumentConverter functionality.
    
    WHAT THIS SHOWS:
    ----------------
    1. How to initialize converter
    2. How to convert a document
    3. How to access results
    4. How errors are handled
    5. What metadata is available
    """
    
    print("\n" + "="*60)
    print("  Document Converter Demo")
    print("="*60 + "\n")
    
    # Initialize converter
    print("📋 Initializing DocumentConverter...")
    converter = DocumentConverter()
    print(f"   ✅ Markdown output: {converter.markdown_dir}")
    print(f"   ✅ Images output: {converter.images_dir}\n")
    
    # Create a sample text file for testing
    sample_file = Path("data/raw/sample_test.txt")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    
    sample_content = """# Sample Document for Testing

## Introduction

This is a **test document** to demonstrate the DocumentConverter.

Key features:
- Converts documents to Markdown
- Extracts images
- Handles multiple formats
- Provides detailed metadata

## Technical Details

The converter uses:
1. Markitdown (primary)
2. PyPDF2 (PDF fallback)
3. python-docx (DOCX fallback)

### Code Example

```python
converter = DocumentConverter()
result = converter.convert("document.pdf")
print(result.markdown)
```

## Conclusion

The converter is production-ready and handles errors gracefully.
"""
    
    sample_file.write_text(sample_content, encoding='utf-8')
    print(f"📝 Created sample file: {sample_file.name}\n")
    
    # Convert the document
    print("🔄 Converting document...")
    result = converter.convert(str(sample_file))
    
    # Display results
    if result.success:
        print("\n✅ CONVERSION SUCCESSFUL!\n")
        
        print("📊 Metadata:")
        for key, value in result.metadata.items():
            print(f"   {key}: {value}")
        
        print(f"\n📄 Markdown Preview (first 300 characters):")
        print("-" * 60)
        print(result.markdown[:300] + "...")
        print("-" * 60)
        
        print(f"\n🖼️  Images extracted: {len(result.images)}")
        
        print(f"\n💾 Saved to: {result.metadata.get('markdown_path')}")
        
    else:
        print(f"\n❌ CONVERSION FAILED!")
        print(f"   Error: {result.error}")
    
    print("\n" + "="*60 + "\n")
    
    # Demonstrate error handling
    print("🧪 Testing error handling with non-existent file...")
    result_error = converter.convert("nonexistent_file.pdf")
    
    if not result_error.success:
        print(f"   ✅ Error caught correctly: {result_error.error}\n")
    
    print("✅ Demo complete!\n")


if __name__ == "__main__":
    demo_converter()
