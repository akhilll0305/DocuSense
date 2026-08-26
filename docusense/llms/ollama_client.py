"""
Ollama LLM Client - Local LLM inference via Ollama.

This is Step 1 of Phase 4: Answer Generation with Citations.

PURPOSE:
--------
Wrapper around the Ollama Python library for local LLM inference:
1. Text generation (prompt → response)
2. Chat-based generation (messages → response)
3. Health checks and model validation
4. Retry logic for reliability
5. Streaming support for future UI integration

WHY OLLAMA?
-----------
- $0 cost (vs GPT-4: $10-30 per 1M tokens)
- Privacy (papers stay local, no API calls)
- No rate limits (unlimited local inference)
- Fast inference with llama3.2:3b (~1-3s per response)

MODELS:
-------
- llama3.2:3b (DEFAULT): Fast, good quality, 3B params
- llama3.1:8b: Better quality, slower
- mistral:7b: Good alternative
- phi3:mini: Very fast, 3.8B params

Author: DocuSense
Created: 2026-03-06
"""

import time
from typing import Optional, List, Dict, Generator, Any

from loguru import logger

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not installed. Run: pip install ollama")

from docusense.config.settings import settings


class OllamaClient:
    """
    Client for Ollama local LLM inference.
    
    Features:
    - Simple generate and chat interfaces
    - Automatic retry on failure
    - Streaming support
    - Model availability checks
    - Configurable temperature and token limits
    
    Usage:
        client = OllamaClient()
        if client.is_available():
            response = client.generate("What is BERT?")
            print(response)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (default from settings: llama3.2:3b)
            base_url: Ollama server URL (default: http://localhost:11434)
            temperature: Generation temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            max_retries: Number of retries on failure
            retry_delay: Delay between retries (seconds)
        """
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens or settings.answer_max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Configure Ollama client host
        if OLLAMA_AVAILABLE:
            self._client = ollama.Client(host=self.base_url)
        else:
            self._client = None
        
        logger.info("🤖 OllamaClient initialized")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  URL: {self.base_url}")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Max tokens: {self.max_tokens}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is running and model is available.
        
        Returns:
            True if Ollama is reachable and model exists
        """
        if not OLLAMA_AVAILABLE or not self._client:
            return False
        
        try:
            models = self._client.list()
            model_names = [m.model for m in models.models]
            
            # Check if our model is available (match with or without tag)
            model_base = self.model.split(":")[0]
            available = any(
                m == self.model or m.startswith(model_base)
                for m in model_names
            )
            
            if available:
                logger.info(f"✅ Ollama available with model: {self.model}")
            else:
                logger.warning(
                    f"⚠️ Model '{self.model}' not found. "
                    f"Available: {model_names}. "
                    f"Run: ollama pull {self.model}"
                )
            
            return available
        except Exception as e:
            logger.warning(f"❌ Ollama not available: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If Ollama is unavailable or generation fails
        """
        if not OLLAMA_AVAILABLE or not self._client:
            raise RuntimeError(
                "Ollama is not available. Install: pip install ollama, "
                "then start Ollama and pull a model: ollama pull llama3.2:3b"
            )
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        
        logger.info(f"📝 Generating response (model: {self.model})")
        logger.debug(f"  Prompt length: {len(prompt)} chars")
        
        # Build options
        options = {
            "temperature": temp,
            "num_predict": tokens,
        }
        
        # Retry logic
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                
                response = self._client.generate(
                    model=self.model,
                    prompt=prompt,
                    system=system_prompt or "",
                    options=options
                )
                
                elapsed = time.time() - start_time
                result = response.response.strip()
                
                logger.success(
                    f"✅ Generated {len(result)} chars in {elapsed:.2f}s "
                    f"(attempt {attempt})"
                )
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"⚠️ Generation failed (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        
        raise RuntimeError(f"Generation failed after {self.max_retries} attempts: {last_error}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Chat-based generation with message history.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Assistant's response text
        """
        if not OLLAMA_AVAILABLE or not self._client:
            raise RuntimeError("Ollama is not available")
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        
        logger.info(f"💬 Chat generation ({len(messages)} messages)")
        
        options = {
            "temperature": temp,
            "num_predict": tokens,
        }
        
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options=options
                )
                
                elapsed = time.time() - start_time
                result = response.message.content.strip()
                
                logger.success(f"✅ Chat response: {len(result)} chars in {elapsed:.2f}s")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Chat failed (attempt {attempt}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        
        raise RuntimeError(f"Chat failed after {self.max_retries} attempts: {last_error}")
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """
        Stream text generation token by token.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Override default temperature
            
        Yields:
            Text chunks as they are generated
        """
        if not OLLAMA_AVAILABLE or not self._client:
            raise RuntimeError("Ollama is not available")
        
        temp = temperature if temperature is not None else self.temperature
        
        logger.info(f"🌊 Streaming generation (model: {self.model})")
        
        options = {"temperature": temp}
        
        try:
            stream = self._client.generate(
                model=self.model,
                prompt=prompt,
                system=system_prompt or "",
                options=options,
                stream=True
            )
            
            for chunk in stream:
                yield chunk.response
                
        except Exception as e:
            logger.error(f"❌ Stream generation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not OLLAMA_AVAILABLE or not self._client:
            return {"error": "Ollama not available"}
        
        try:
            info = self._client.show(self.model)
            return {
                "model": self.model,
                "base_url": self.base_url,
                "parameters": info.get("parameters", ""),
                "template": info.get("template", "")[:200],
            }
        except Exception as e:
            return {"error": str(e)}
