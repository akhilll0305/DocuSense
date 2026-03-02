"""
Configuration settings for DocuSense.

Centralized configuration management using pydantic-settings.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==================== Project Info ====================
    project_name: str = "DocuSense"
    version: str = "0.1.0"
    environment: Literal["dev", "test", "prod"] = "dev"
    
    # ==================== LLM Providers ====================
    # OpenAI (optional, for comparison/experimentation only)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # HuggingFace (FREE)
    huggingface_api_key: Optional[str] = None  # Free tier available
    
    # Gemini (FREE tier - for image processing & query rewriting)
    gemini_api_key: Optional[str] = None  # FREE: 1500 images/day, 15 req/min
    gemini_model: str = "gemini-2.0-flash"  # Fast, multimodal (vision + text)
    
    # Ollama (FREE - local models)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"  # Free local model
    ollama_vision_model: str = "llava:7b"  # Free vision model (backup)
    
    # Default LLM models (FREE OPTIONS)
    default_llm_provider: str = "ollama"  # FREE: Ollama local models
    default_chat_model: str = "llama3.2:3b"  # FREE: Fast, good quality
    default_smart_model: str = "llama3.2:3b"  # FREE: Can upgrade to llama3.1:8b for better quality
    
    # ==================== Embedding Settings (Phase 2) ====================
    # FREE OPTIONS ONLY
    embedding_provider: str = "sentence-transformers"  # FREE: Local embeddings
    embedding_model: str = "all-MiniLM-L6-v2"  # FREE: Fast, 384 dim, good quality
    # Alternative models:
    # - "all-mpnet-base-v2" (768 dim, better quality, slower)
    # - "paraphrase-MiniLM-L6-v2" (384 dim, optimized for semantic similarity)
    # - "multi-qa-MiniLM-L6-cos-v1" (384 dim, optimized for Q&A)
    embedding_dimension: int = 384  # all-MiniLM-L6-v2
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"  # "cuda" if GPU available
    embedding_normalize: bool = True  # Normalize vectors for cosine similarity
    
    # ==================== Document Processing (Phase 1) ====================
    # File handling
    allowed_file_types: list[str] = [".pdf", ".docx", ".txt", ".md", ".pptx", ".xlsx"]
    max_file_size_mb: int = 100  # Maximum file size in MB
    
    # Document conversion
    convert_to_markdown: bool = True  # Always convert to Markdown first
    preserve_images: bool = True  # Extract and process images
    
    # Image processing
    use_image_understanding: bool = True  # Use vision LLM for images
    image_vision_provider: Literal["gemini", "llava", "ocr-only"] = "gemini"
    gemini_rate_limit_per_min: int = 15  # FREE tier limit
    image_fallback_to_ocr: bool = True  # Use OCR if vision fails
    max_image_size_mb: int = 5  # Resize if larger
    
    # ==================== Chunking Settings ====================
    # Token limits
    min_chunk_tokens: int = 200  # Minimum chunk size
    max_chunk_tokens: int = 800  # Maximum chunk size
    target_chunk_tokens: int = 500  # Target chunk size
    embedding_token_limit: int = 512  # Hard limit for embedding model
    
    # Chunking strategy
    chunk_size: int = 500  # Legacy - use target_chunk_tokens
    chunk_overlap: int = 50
    chunking_strategy: Literal["fixed", "semantic", "sliding"] = "semantic"
    
    # Semantic chunking (Markdown-aware)
    split_on_headers: bool = True  # Split at ## level
    preserve_code_blocks: bool = True  # Keep code together
    preserve_tables: bool = True  # Keep tables together
    overlap_strategy: Literal["token", "sentence"] = "sentence"
    
    # ==================== Vector Store Settings (Phase 2) ====================
    
    # Qdrant configuration
    vector_store_type: Literal["qdrant", "faiss", "chroma"] = "qdrant"
    qdrant_mode: Literal["memory", "disk", "server"] = "disk"  # memory=testing, disk=local, server=remote
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"  # For disk mode
    qdrant_url: Optional[str] = None  # For server mode: http://localhost:6333
    qdrant_api_key: Optional[str] = None  # For cloud Qdrant
    qdrant_collection_name: str = "docusense_chunks"
    
    # Distance metric for similarity evaluation
    distance_metric: Literal["COSINE", "EUCLIDEAN", "DOT"] = "COSINE"
    
    # Vector search
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    use_score_threshold: bool = True  # Filter by similarity score
    
    # Auto-detect server mode when credentials are provided
    @property
    def effective_qdrant_mode(self) -> str:
        """Return 'server' mode if credentials provided, else use configured mode."""
        if self.qdrant_url and self.qdrant_api_key:
            return "server"
        return self.qdrant_mode
    
    # Hybrid retrieval
    use_hybrid_search: bool = True
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    
    # Re-ranking (FREE)
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # FREE
    rerank_top_k: int = 20  # Retrieve 20, rerank to top_k
    
    # ==================== Query Processing ====================
    enable_query_rewriting: bool = True
    enable_intent_classification: bool = True
    enable_strategy_planning: bool = True
    
    # ==================== Answer Generation ====================
    max_context_tokens: int = 3000
    answer_max_tokens: int = 500
    temperature: float = 0.0
    include_citations: bool = True
    
    # ==================== System Settings ====================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str = "logs/docusense.log"
    log_rotation: str = "500 MB"
    log_retention: str = "10 days"
    
    cache_enabled: bool = True
    cache_dir: str = "data/.cache"
    
    # ==================== Paths ====================
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    markdown_data_dir: Path = processed_data_dir / "markdown"  # Phase 1: Converted Markdown
    images_data_dir: Path = processed_data_dir / "images"  # Phase 1: Extracted images
    vector_store_dir: Path = data_dir / "vector_stores"
    logs_dir: Path = PROJECT_ROOT / "logs"
    
    # Database
    sqlite_db_path: Path = data_dir / "docusense.db"  # Phase 1: Chunks storage
    
    # ==================== API Settings (FastAPI) ====================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True  # Auto-reload in dev
    api_title: str = "DocuSense API"
    api_description: str = "Intelligent Document Q&A using RAG"
    
    # ==================== UI Settings (Gradio) ====================
    ui_share: bool = False  # Create shareable link
    ui_server_name: str = "0.0.0.0"
    ui_server_port: int = 7860
    
    # ==================== Deployment (Modal.com) ====================
    modal_deployment: bool = False
    modal_gpu: str = "any"  # "any", "a10g", "a100", etc.
    modal_cpu: int = 2
    modal_memory: int = 4096  # MB
    modal_timeout: int = 600  # seconds
    
    # ==================== Evaluation ====================
    golden_dataset_path: str = "data/evaluation/golden_queries.json"
    enable_query_logging: bool = True
    
    # ==================== Rate Limiting ====================
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_data_dir.mkdir(parents=True, exist_ok=True)
        self.images_data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        if self.cache_enabled:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "prod"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "dev"
    
    def get_llm_config(self, task: str = "default") -> dict:
        """Get LLM configuration for specific task (FREE models only)."""
        task_models = {
            "planning": self.default_smart_model,
            "classification": self.default_chat_model,
            "synthesis": self.default_smart_model,
            "reranking": self.default_chat_model,
            "default": self.default_chat_model
        }
        
        return {
            "provider": self.default_llm_provider,
            "model": task_models.get(task, self.default_chat_model),
            "temperature": self.temperature,
            "max_tokens": self.answer_max_tokens,
            "timeout": self.timeout,
            "base_url": self.ollama_base_url if self.default_llm_provider == "ollama" else None
        }
    
    @property
    def is_using_free_models(self) -> bool:
        """Check if using 100% free models."""
        return (
            self.default_llm_provider in ["ollama", "huggingface"] and
            self.embedding_provider == "sentence-transformers"
        )
    
    @property
    def deployment_ready(self) -> bool:
        """Check if configuration is ready for modal.com deployment."""
        # Modal.com deployment requires no external API dependencies for free tier
        return self.is_using_free_models


# Global settings instance
settings = Settings()
