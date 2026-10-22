"""
DocuSense API Module.

Run: uvicorn docusense.api.app:app --reload
Docs: http://localhost:8000/docs
"""

from .app import app
from .routes import router
from .schemas import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    IngestResponse,
    HealthResponse,
)

__all__ = [
    "AskRequest",
    "AskResponse",
    "ChatRequest",
    "ChatResponse",
    "HealthResponse",
    "IngestResponse",
    "app",
    "router",
]
