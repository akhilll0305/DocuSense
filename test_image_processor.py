"""
Test script for Image Processor.

Demonstrates vision model capabilities with sample images.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from docusense.ingestion.image_processor import ImageProcessor, VisionProvider
from loguru import logger

# Configure logging
logger.add("logs/image_processor_demo.log", rotation="10 MB")


def create_sample_image() -> Path:
    """
    Create a sample image with text and shapes for testing.
    
    This simulates what might be extracted from a document.
    """
    # Create image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw title
    draw.text((50, 50), "Q4 2024 Revenue Report", fill='black')
    
    # Draw simple bar chart
    bars = [
        ("Q1", 150, 'blue'),
        ("Q2", 220, 'blue'),
        ("Q3", 280, 'blue'),
        ("Q4", 380, 'green')
    ]
    
    x_start = 100
    for i, (label, height, color) in enumerate(bars):
        x = x_start + i * 150
        y = 500 - height
        
        # Draw bar
        draw.rectangle([x, y, x + 80, 500], fill=color, outline='black')
        
        # Draw label
        draw.text((x + 10, 510), label, fill='black')
        
        # Draw value
        value = f"${height}M"
        draw.text((x + 10, y - 30), value, fill='black')
    
    # Draw axis labels
    draw.text((50, 520), "Quarter", fill='black')
    draw.text((30, 250), "Revenue ($M)", fill='black')
    
    # Save image
    image_path = Path("data/processed/images/sample_chart.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)
    
    return image_path


def demo_image_processor():
    """
    Demonstrate ImageProcessor functionality.
    
    WHAT THIS SHOWS:
    ----------------
    1. How to initialize image processor
    2. Available vision providers
    3. Processing an image with context
    4. Accessing description results
    5. Rate limit monitoring
    """
    print("\n" + "="*70)
    print("  Image Processor Demo")
    print("="*70 + "\n")
    
    # Step 1: Create sample image
    print("📊 Creating sample chart image...")
    image_path = create_sample_image()
    print(f"   ✅ Created: {image_path}\n")
    
    # Step 2: Initialize processor
    print("🔧 Initializing ImageProcessor...")
    processor = ImageProcessor()
    print()
    
    # Step 3: Check available providers
    print("📋 Available Vision Providers:")
    if processor.gemini_client:
        print("   ✅ Gemini 2.0 Flash (15 images/min FREE)")
    else:
        print("   ⚠️ Gemini not available (no API key)")
    
    if processor.llava_available:
        print("   ✅ LLaVA via Ollama (unlimited FREE)")
    else:
        print("   ⚠️ LLaVA not available (ollama pull llava:7b)")
    
    if processor._extract_ocr_text(Image.new('RGB', (100, 100))) is not None:
        print("   ✅ Tesseract OCR (unlimited FREE)")
    else:
        print("   ⚠️ Tesseract OCR not available")
    
    print()
    
    # Step 4: Process image with context
    if not processor.gemini_client and not processor.llava_available:
        print("⚠️ No vision models available!")
        print("   To use Gemini: Set GEMINI_API_KEY in .env")
        print("   To use LLaVA: Run 'ollama pull llava:7b'")
        print()
        return
    
    print("🖼️  Processing image with context...")
    
    context = """
    This chart appears in the Financial Results section of our Q4 2024 report.
    It shows quarterly revenue growth throughout the year.
    """
    
    result = processor.process_image(
        str(image_path),
        context=context.strip()
    )
    
    # Step 5: Display results
    if result.success:
        print("\n✅ IMAGE PROCESSING SUCCESSFUL!\n")
        
        print("📊 Results:")
        print(f"   Provider: {result.provider.upper()}")
        print(f"   Confidence: {result.confidence:.1%}")
        print()
        
        print("📝 Description:")
        print("-" * 70)
        print(result.description)
        print("-" * 70)
        print()
        
        if result.ocr_text:
            print("🔤 OCR Text Extracted:")
            print("-" * 70)
            print(result.ocr_text[:200] + "..." if len(result.ocr_text) > 200 else result.ocr_text)
            print("-" * 70)
            print()
        
        # Step 6: Show rate limit status
        if result.provider == "gemini":
            status = processor.get_rate_limit_status()
            print("⏱️  Rate Limit Status:")
            print(f"   Used: {status['count']}/{status['limit']} requests")
            print(f"   Remaining: {status['remaining']}")
            print(f"   Resets in: {status['resets_in_seconds']:.0f} seconds")
            print()
    
    else:
        print(f"\n❌ IMAGE PROCESSING FAILED!")
        print(f"   Error: {result.error}\n")
    
    print("="*70 + "\n")
    
    # Step 7: Demonstrate how this enriches document conversion
    print("💡 How This Works in Document Conversion:\n")
    print("1. Document converted to Markdown (Step 3 ✅)")
    print("2. Images extracted from document")
    print("3. Vision model describes each image ← YOU ARE HERE")
    print("4. Descriptions injected into Markdown:")
    print()
    print("   BEFORE:")
    print("   ```markdown")
    print("   ## Financial Results")
    print("   ![chart](image_001.png)")
    print("   Our revenue grew significantly...")
    print("   ```")
    print()
    print("   AFTER:")
    print("   ```markdown")
    print("   ## Financial Results")
    print("   ![chart](image_001.png)")
    print("   **Image Description:** Bar chart showing quarterly revenue")
    print("   from Q1 ($150M) to Q4 ($380M), demonstrating 153% growth.")
    print()
    print("   Our revenue grew significantly...")
    print("   ```")
    print()
    print("5. Chunked with descriptions (Step 6)")
    print("6. Embedded & searchable (Phase 2)")
    print()
    print("✅ Demo complete!\n")


if __name__ == "__main__":
    demo_image_processor()
