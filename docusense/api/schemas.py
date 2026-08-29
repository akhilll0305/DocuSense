"""
API Schemas - Request/Response models for the DocuSense API.

Phase 7: API & UI (Step 1)

Pydantic models for all API endpoints.

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# INGESTION
# ==============================================================================

class IngestResponse(BaseModel):
    """Response from document ingestion."""
    success: bool
    document_id: str
    filename: str
    num_chunks: int = 0
    num_embeddings: int = 0
    is_research_paper: bool = False
    paper_title: Optional[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None


class BatchIngestResponse(BaseModel):
    """Response from batch ingestion."""
    total: int
    successful: int
    failed: int
    results: List[IngestResponse]


# ==============================================================================
# QUESTION ANSWERING
# ==============================================================================

class AskRequest(BaseModel):
    """Request to ask a question."""
    query: str = Field(..., min_length=1, description="The question to ask")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    mode: str = Field("answer", description="Mode: answer, compare, conflicts")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class AskResponse(BaseModel):
    """Response to a question."""
    answer: str
    sources: List[Dict[str, Any]] = []
    papers_cited: List[str] = []
    reference_list: str = ""
    confidence: float = 0.0
    has_citations: bool = False
    num_sources: int = 0
    total_time: float = 0.0


# ==============================================================================
# CHAT (Multi-turn)
# ==============================================================================

class StartChatRequest(BaseModel):
    """Request to start a new chat."""
    title: str = Field("New Chat", description="Chat title")


class StartChatResponse(BaseModel):
    """Response with conversation ID."""
    conversation_id: str
    title: str


class ChatRequest(BaseModel):
    """Request for a chat turn."""
    query: str = Field(..., min_length=1, description="Your question")
    mode: str = Field("answer", description="Mode: answer, compare, conflicts")
    top_k: int = Field(5, ge=1, le=20)


class ChatResponse(BaseModel):
    """Response from a chat turn."""
    answer: str
    conversation_id: str
    message_id: str
    sources: List[Dict[str, Any]] = []
    papers_cited: List[str] = []
    reference_list: str = ""
    confidence: float = 0.0
    response_time: float = 0.0
    turn_number: int = 0


class ChatHistoryResponse(BaseModel):
    """Chat history response."""
    conversation_id: str
    messages: List[Dict[str, Any]]


class ConversationListItem(BaseModel):
    """A conversation in the list."""
    conversation_id: str
    title: str
    created_at: str
    updated_at: str


# ==============================================================================
# DOCUMENTS
# ==============================================================================

class DocumentInfo(BaseModel):
    """Document information."""
    document_id: str
    filename: str
    file_type: str = ""
    total_chunks: int = 0
    processing_date: str = ""
    is_research_paper: bool = False
    paper_title: str = ""


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: List[DocumentInfo]
    total: int


# ==============================================================================
# SYSTEM
# ==============================================================================

class HealthResponse(BaseModel):
    """System health check."""
    status: str = "healthy"
    version: str = "1.0.0"
    components: Dict[str, bool] = {}
    # Which backend writes the answers, and therefore whether the text of a
    # document leaves this machine. The landing and sign-in pages both claim
    # it does not, which is true of a local install and false of a deployment
    # using a hosted model, so they read this rather than assert it.
    generation: str = "local"


class StatsResponse(BaseModel):
    """Query statistics."""
    total_queries: int = 0
    avg_response_time: float = 0.0
    avg_confidence: float = 0.0
    avg_results: float = 0.0
