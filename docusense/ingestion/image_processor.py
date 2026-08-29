"""
Image Processor - Extracts meaning from images using vision models.

This module is a CRITICAL component of Phase 1 that ensures we don't lose
information stored in images (charts, diagrams, screenshots, tables).

PURPOSE:
--------
Convert images to text descriptions that can be:
1. Embedded as vectors (semantic search)
2. Read by the LLM during answer generation
3. Cited in responses ("As shown in Figure 2...")

WHY IMAGE UNDERSTANDING MATTERS:
--------------------------------
Documents contain crucial information in visual form:
- Charts/graphs: "Revenue increased 45% Q4 2024"
- Diagrams: "System architecture with 3-tier design"
- Screenshots: "Error message showing timeout at line 42"
- Tables: "Monthly sales data Jan-Dec"

Without image processing, we'd LOSE all this information!

MULTI-TIER STRATEGY (FREE):
---------------------------
Tier 1: Gemini 2.0 Flash Vision (PRIMARY)
  - Fast, accurate, multimodal
  - FREE: 1500 images/day, 15 req/min
  - Understands charts, diagrams, code, tables
  - 2-3 seconds per image

Tier 2: LLaVA via Ollama (BACKUP)
  - Completely FREE, unlimited
  - Runs locally (no API calls)
  - 4-6 seconds per image
  - Good quality, privacy-preserving

Tier 3: Tesseract OCR (FALLBACK)
  - Text extraction only
  - FREE, unlimited
  - Fast (~1 second)
  - Good for text-heavy images

RATE LIMITING:
--------------
Gemini free tier: 15 requests/minute
Strategy: Track usage, auto-switch to LLaVA after 15 images/min
"""

import time
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
from enum import Enum

from loguru import logger
from PIL import Image
import ollama

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available - OCR fallback disabled")


from docusense.config import settings


def _import_genai():
    """Import the Gemini SDK on first use; see query_processor for why."""
    import google.generativeai as genai

    return genai


def _genai_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("google.generativeai") is not None


GEMINI_AVAILABLE = _genai_available()


class VisionProvider(Enum):
    """Available vision model providers."""
    GEMINI = "gemini"
    LLAVA = "llava"
    OCR = "ocr"


@dataclass
class ImageDescription:
    """
    Result of image understanding.
    
    Attributes:
        success: Whether description was generated
        description: Human-readable image description
        ocr_text: Text extracted from image (if any)
        provider: Which model was used (gemini/llava/ocr)
        confidence: How confident we are (0-1)
        error: Error message if failed
    """
    success: bool
    description: str
    ocr_text: Optional[str] = None
    provider: Optional[str] = None
    confidence: float = 1.0
    error: Optional[str] = None


class RateLimiter:
    """
    Simple rate limiter for API calls.
    
    Tracks requests per minute and blocks if limit exceeded.
    Resets counter every 60 seconds.
    
    Example:
        >>> limiter = RateLimiter(limit=15)
        >>> if limiter.can_proceed():
        >>>     limiter.increment()
        >>>     make_api_call()
        >>> else:
        >>>     print("Rate limit reached!")
    """
    
    def __init__(self, limit: int = 15):
        """
        Initialize rate limiter.
        
        Args:
            limit: Maximum requests per minute
        """
        self.limit = limit
        self.count = 0
        self.window_start = time.time()
    
    def can_proceed(self) -> bool:
        """
        Check if we can make another request.
        
        Returns:
            True if under limit, False if rate limited
        """
        self._reset_if_needed()
        return self.count < self.limit
    
    def increment(self):
        """Increment request counter."""
        self._reset_if_needed()
        self.count += 1
    
    def _reset_if_needed(self):
        """Reset counter if 60 seconds elapsed."""
        now = time.time()
        if now - self.window_start >= 60:
            self.count = 0
            self.window_start = now
    
    def get_status(self) -> Dict[str, any]:
        """Get current rate limit status."""
        self._reset_if_needed()
        return {
            "count": self.count,
            "limit": self.limit,
            "remaining": self.limit - self.count,
            "resets_in_seconds": 60 - (time.time() - self.window_start)
        }


class ImageProcessor:
    """
    Process images using vision models to extract semantic meaning.
    
    ARCHITECTURE:
    -------------
    1. Load image from disk
    2. Resize if too large (API limits)
    3. Try primary provider (Gemini)
    4. If rate limited/failed, try LLaVA
    5. If all vision fails, try OCR
    6. Return rich description
    
    EXAMPLE USAGE:
    --------------
    >>> processor = ImageProcessor()
    >>> 
    >>> # Process single image
    >>> result = processor.process_image(
    >>>     "chart.png",
    >>>     context="This appears in a financial report section"
    >>> )
    >>> 
    >>> if result.success:
    >>>     print(result.description)
    >>>     print(f"Provider: {result.provider}")
    >>>     if result.ocr_text:
    >>>         print(f"Text found: {result.ocr_text}")
    """
    
    def __init__(self):
        """Initialize image processor with all available providers."""
        self.gemini_limiter = RateLimiter(limit=settings.gemini_rate_limit_per_min)
        self.gemini_client = None
        
        # Initialize Gemini if API key available
        if GEMINI_AVAILABLE and settings.gemini_api_key:
            try:
                _import_genai().configure(api_key=settings.gemini_api_key)
                self.gemini_client = _import_genai().GenerativeModel(settings.gemini_model)
                logger.info(f"✅ Gemini vision initialized: {settings.gemini_model}")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
                self.gemini_client = None
        
        # Check LLaVA availability
        self.llava_available = self._check_llava()
        
        # Log available providers
        providers = []
        if self.gemini_client:
            providers.append("Gemini (15/min)")
        if self.llava_available:
            providers.append("LLaVA (unlimited)")
        if TESSERACT_AVAILABLE:
            providers.append("OCR (unlimited)")
        
        logger.info(f"Image processing available: {', '.join(providers) if providers else 'NONE'}")
    
    def _check_llava(self) -> bool:
        """
        Check if LLaVA is available via Ollama.
        
        Returns:
            True if LLaVA model is pulled and ready
        """
        try:
            models = ollama.list()
            llava_models = [m for m in models.get('models', []) 
                           if 'llava' in m.get('name', '').lower()]
            
            if llava_models:
                logger.info("✅ LLaVA vision available via Ollama")
                return True
            else:
                logger.warning("⚠️ LLaVA not found. Install with: ollama pull llava:7b")
                return False
                
        except Exception as e:
            logger.debug(f"LLaVA check failed: {e}")
            return False
    
    def process_image(
        self, 
        image_path: str, 
        context: str = "",
        force_provider: Optional[VisionProvider] = None
    ) -> ImageDescription:
        """
        Process image to extract semantic description.
        
        PROCESS:
        --------
        1. Load and validate image
        2. Resize if needed (API limits)
        3. Try vision models (Gemini → LLaVA)
        4. Extract OCR text if enabled
        5. Combine results into rich description
        
        Args:
            image_path: Path to image file
            context: Surrounding text context (helps vision model)
            force_provider: Force specific provider (for testing)
            
        Returns:
            ImageDescription with description and metadata
            
        Example:
            >>> processor = ImageProcessor()
            >>> result = processor.process_image(
            >>>     "revenue_chart.png",
            >>>     context="Q4 financial results section"
            >>> )
            >>> print(result.description)
            # "Bar chart showing quarterly revenue growth from $1.2M to $2.5M"
        """
        image_path = Path(image_path)
        
        # Validate image exists
        if not image_path.exists():
            return ImageDescription(
                success=False,
                description="",
                error=f"Image not found: {image_path}"
            )
        
        logger.info(f"Processing image: {image_path.name}")
        
        # Load and prepare image
        try:
            image = Image.open(image_path)
            image = self._prepare_image(image)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return ImageDescription(
                success=False,
                description="",
                error=f"Image load failed: {e}"
            )
        
        # Try vision models in priority order
        description = None
        provider = None
        
        if force_provider:
            # Use specific provider (for testing)
            if force_provider == VisionProvider.GEMINI:
                description = self._describe_with_gemini(image, context)
                provider = "gemini"
            elif force_provider == VisionProvider.LLAVA:
                description = self._describe_with_llava(image_path, context)
                provider = "llava"
        else:
            # Auto-select based on availability and rate limits
            if self.gemini_client and self.gemini_limiter.can_proceed():
                # Try Gemini (fast, accurate, rate limited)
                description = self._describe_with_gemini(image, context)
                if description:
                    provider = "gemini"
                    self.gemini_limiter.increment()
            
            if not description and self.llava_available:
                # Fallback to LLaVA (slower, unlimited)
                logger.info("Using LLaVA fallback")
                description = self._describe_with_llava(image_path, context)
                if description:
                    provider = "llava"
        
        # Extract OCR text (always try, supplements vision)
        ocr_text = None
        if TESSERACT_AVAILABLE and settings.image_fallback_to_ocr:
            ocr_text = self._extract_ocr_text(image)
        
        # If all vision failed but we have OCR, use that
        if not description and ocr_text:
            description = f"Text extracted from image: {ocr_text}"
            provider = "ocr"
        
        # Build final description
        if description or ocr_text:
            final_description = description or ""
            if ocr_text and provider != "ocr":
                # Append OCR text to vision description
                final_description += f"\n\nText in image: {ocr_text}"
            
            logger.success(f"✅ Image described via {provider}: {len(final_description)} chars")
            
            return ImageDescription(
                success=True,
                description=final_description.strip(),
                ocr_text=ocr_text,
                provider=provider,
                confidence=1.0 if provider in ["gemini", "llava"] else 0.7
            )
        else:
            logger.warning("❌ All image processing methods failed")
            return ImageDescription(
                success=False,
                description="",
                error="No vision models or OCR available"
            )
    
    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """
        Prepare image for processing.
        
        OPTIMIZATIONS:
        --------------
        - Resize if > 5 MB (API efficiency)
        - Convert to RGB if needed (remove alpha channel)
        - Maintain aspect ratio
        
        Args:
            image: PIL Image object
            
        Returns:
            Processed PIL Image
        """
        # Convert to RGB if needed (remove transparency)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        max_dimension = 2048  # Reasonable for API + quality
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {new_size}")
        
        return image
    
    def _describe_with_gemini(self, image: Image.Image, context: str) -> Optional[str]:
        """
        Describe image using Gemini 2.0 Flash vision model.
        
        PROMPT ENGINEERING:
        -------------------
        We provide context to help the model understand:
        - What document section this is from
        - What type of information to focus on
        - How detailed to be
        
        Args:
            image: PIL Image object
            context: Surrounding text context
            
        Returns:
            Description string or None if failed
        """
        if not self.gemini_client:
            return None
        
        try:
            # Build context-aware prompt
            prompt = self._build_vision_prompt(context)
            
            # Generate description (Gemini accepts PIL Image directly)
            response = self.gemini_client.generate_content([prompt, image])
            
            description = response.text.strip()
            
            logger.debug(f"Gemini description: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.warning(f"Gemini vision failed: {e}")
            return None
    
    def _describe_with_llava(self, image_path: Path, context: str) -> Optional[str]:
        """
        Describe image using LLaVA via Ollama.
        
        LLaVA (Large Language and Vision Assistant):
        - Open-source vision model
        - Runs locally via Ollama
        - Unlimited, free, private
        - Slower than Gemini (~5s vs ~2s)
        
        Args:
            image_path: Path to image file
            context: Surrounding text context
            
        Returns:
            Description string or None if failed
        """
        if not self.llava_available:
            return None
        
        try:
            prompt = self._build_vision_prompt(context)
            
            # Call LLaVA through Ollama
            response = ollama.generate(
                model=settings.ollama_vision_model,
                prompt=prompt,
                images=[str(image_path)]
            )
            
            description = response.get('response', '').strip()
            
            logger.debug(f"LLaVA description: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.warning(f"LLaVA vision failed: {e}")
            return None
    
    def _extract_ocr_text(self, image: Image.Image) -> Optional[str]:
        """
        Extract text from image using Tesseract OCR.
        
        WHEN USEFUL:
        ------------
        - Screenshots with text
        - Scanned documents
        - Diagrams with labels
        - Tables with text data
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text or None
        """
        if not TESSERACT_AVAILABLE:
            return None
        
        try:
            text = pytesseract.image_to_string(image).strip()
            
            if text and len(text) > 10:  # Ignore noise
                logger.debug(f"OCR extracted: {len(text)} chars")
                return text
            
            return None
            
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return None
    
    def _build_vision_prompt(self, context: str) -> str:
        """
        Build context-aware prompt for vision models.
        
        PROMPT ENGINEERING:
        -------------------
        Good prompts make vision models more accurate and relevant.
        
        We include:
        - Task description
        - Context from surrounding text
        - What to focus on
        - Output format
        
        Args:
            context: Surrounding text from document
            
        Returns:
            Optimized prompt string
        """
        base_prompt = """Describe this image from a document in detail. Focus on:

1. **Type of visual**: Is it a chart, diagram, screenshot, photo, table, or code?
2. **Key information**: What data, concepts, or information does it convey?
3. **Specific details**: Numbers, labels, text, relationships, trends
4. **Relevance**: How does it relate to the document content?

Be specific and factual. If there's text, include it. If there are numbers or data, mention them."""

        if context:
            base_prompt += f"\n\n**Context from document**: {context[:200]}"
        
        return base_prompt
    
    def get_rate_limit_status(self) -> Dict[str, any]:
        """
        Get current rate limit status for monitoring.
        
        Returns:
            Dictionary with rate limit info
        """
        status = self.gemini_limiter.get_status()
        status['provider'] = 'gemini'
        return status
