"""
API Routes - FastAPI endpoints for DocuSense.

Phase 7: API & UI (Step 2)

Endpoints:
----------
POST /api/ingest         - Upload and ingest a document
POST /api/ask            - Ask a question
POST /api/chat/{id}      - Chat in a conversation
POST /api/chat/start     - Start a new chat
GET  /api/chat/{id}      - Get chat history
GET  /api/chats          - List conversations
GET  /api/documents      - List documents
DELETE /api/documents/{id} - Delete a document
GET  /api/health         - Health check
GET  /api/stats          - Query statistics

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from loguru import logger

from docusense.api.schemas import (
    IngestResponse,
    AskRequest,
    AskResponse,
    StartChatRequest,
    StartChatResponse,
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    ConversationListItem,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    StatsResponse,
)

# Global RAG instance (initialized in app.py)
_rag_instance = None


def get_rag():
    """Dependency to get the RAG instance."""
    if _rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return _rag_instance


def set_rag_instance(rag):
    """Set the global RAG instance."""
    global _rag_instance
    _rag_instance = rag


router = APIRouter(prefix="/api", tags=["DocuSense"])


# ==============================================================================
# INGESTION
# ==============================================================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    rag=Depends(get_rag)
):
    """
    Upload and ingest a document (PDF, DOCX, TXT).

    The document goes through: conversion → chunking → embedding → Qdrant storage.
    """
    logger.info(f"📥 API: Ingesting {file.filename}")

    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = rag.ingest(tmp_path)

        return IngestResponse(
            success=result.success,
            document_id=result.document_id,
            filename=file.filename or "unknown",
            num_chunks=result.num_chunks,
            num_embeddings=result.num_embeddings,
            is_research_paper=result.is_research_paper,
            paper_title=result.paper_title,
            processing_time=result.processing_time,
            error=result.error,
        )
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ==============================================================================
# QUESTION ANSWERING
# ==============================================================================

@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    rag=Depends(get_rag)
):
    """
    Ask a question and get an answer with citations.

    Modes:
    - answer: Standard Q&A with citations
    - compare: Multi-paper comparison
    - conflicts: Detect conflicting findings
    """
    logger.info(f"❓ API: '{request.query}' (mode={request.mode})")

    try:
        response = rag.ask(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            mode=request.mode,
        )

        return AskResponse(
            answer=response.answer,
            sources=response.sources,
            papers_cited=response.papers_cited,
            reference_list=response.reference_list,
            confidence=response.confidence,
            has_citations=response.has_citations,
            num_sources=response.num_sources,
            total_time=response.total_time,
        )
    except Exception as e:
        logger.error(f"❌ Ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# MULTI-TURN CHAT
# ==============================================================================

@router.post("/chat/start", response_model=StartChatResponse)
async def start_chat(
    request: StartChatRequest,
    rag=Depends(get_rag)
):
    """Start a new conversation."""
    conv_id = rag.start_chat(request.title)
    return StartChatResponse(conversation_id=conv_id, title=request.title)


@router.post("/chat/{conversation_id}", response_model=ChatResponse)
async def chat(
    conversation_id: str,
    request: ChatRequest,
    rag=Depends(get_rag)
):
    """Send a message in a conversation."""
    logger.info(f"💬 API: Chat {conversation_id}: '{request.query}'")

    try:
        response = rag.chat(
            conversation_id,
            request.query,
            mode=request.mode,
            top_k=request.top_k,
        )

        return ChatResponse(
            answer=response.answer,
            conversation_id=response.conversation_id,
            message_id=response.message_id,
            sources=response.sources,
            papers_cited=response.papers_cited,
            reference_list=response.reference_list,
            confidence=response.confidence,
            response_time=response.response_time,
            turn_number=response.turn_number,
        )
    except Exception as e:
        logger.error(f"❌ Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/{conversation_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    conversation_id: str,
    rag=Depends(get_rag)
):
    """Get all messages in a conversation."""
    messages = rag.get_chat_history(conversation_id)
    return ChatHistoryResponse(
        conversation_id=conversation_id,
        messages=[
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "sources": m.sources,
            }
            for m in messages
        ]
    )


@router.get("/chats", response_model=list[ConversationListItem])
async def list_chats(rag=Depends(get_rag)):
    """List recent conversations."""
    convs = rag.list_chats()
    return [
        ConversationListItem(
            conversation_id=c.conversation_id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in convs
    ]


# ==============================================================================
# DOCUMENT MANAGEMENT
# ==============================================================================

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(rag=Depends(get_rag)):
    """List all ingested documents."""
    docs = rag.list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs)
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    rag=Depends(get_rag)
):
    """Delete a document and its vectors."""
    deleted = rag.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Document {document_id} deleted", "success": True}


# ==============================================================================
# SYSTEM
# ==============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check(rag=Depends(get_rag)):
    """System health check."""
    status = rag.get_status()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components=status.get("components", {})
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(rag=Depends(get_rag)):
    """Get query statistics."""
    status = rag.get_status()
    stats = status.get("query_stats", {})
    return StatsResponse(**stats) if stats else StatsResponse()
