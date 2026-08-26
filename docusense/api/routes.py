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

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from loguru import logger

from docusense.api.deps import get_current_user, get_rag
from docusense.auth import User

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

router = APIRouter(prefix="/api", tags=["DocuSense"])


# ==============================================================================
# SERVER-SENT EVENTS
# ==============================================================================

def _sse_response(make_stream) -> StreamingResponse:
    """
    Wrap a (kind, payload) generator as a Server-Sent Events response.

    Args:
        make_stream: Zero-arg callable returning the generator. Deferred so any
            exception raised while starting it is reported inside the stream
            rather than escaping as an unhandled error mid-response.

    Event payloads:
        {"type": "status", "message": str}
        {"type": "token",  "text": str}
        {"type": "done",   answer, sources, citations, metrics...}
        {"type": "error",  "message": str}
    """
    def encode(kind: str, payload) -> str:
        if kind == "done":
            data = {
                "type": "done",
                "answer": payload.answer,
                "sources": payload.sources,
                "papers_cited": payload.papers_cited,
                "reference_list": payload.reference_list,
                "confidence": payload.confidence,
                "num_sources": getattr(payload, "num_sources", len(payload.sources)),
                "total_time": getattr(payload, "total_time", None)
                or getattr(payload, "response_time", 0.0),
                # Chat turns carry these; a bare ask does not.
                "conversation_id": getattr(payload, "conversation_id", None),
                "message_id": getattr(payload, "message_id", None),
                "turn_number": getattr(payload, "turn_number", None),
                "has_citations": getattr(payload, "has_citations", False),
            }
        elif kind == "token":
            data = {"type": "token", "text": payload}
        else:
            data = {"type": kind, "message": payload}
        return f"data: {json.dumps(data)}\n\n"

    def event_stream():
        try:
            for kind, payload in make_stream():
                yield encode(kind, payload)
        except Exception as e:
            # The response has already begun, so an HTTP error status is no
            # longer possible; report the failure inside the stream instead.
            logger.error(f"❌ Stream failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop nginx buffering the stream
        },
    )


# ==============================================================================
# INGESTION
# ==============================================================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
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
        result = rag.ingest(tmp_path, user_id=user.user_id)

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
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
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
            user_id=user.user_id,
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


@router.post("/ask/stream")
async def ask_question_stream(
    request: AskRequest,
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
):
    """
    Ask a question and receive the answer as Server-Sent Events.

    Generation on a local model takes tens of seconds, so streaming lets the UI
    show text as it is written instead of waiting for the whole response.

    Event stream:
        {"type": "status", "message": str}   progress before text arrives
        {"type": "token",  "text": str}      answer fragments, in order
        {"type": "done",   ...}              sources, citations, metrics
        {"type": "error",  "message": str}   terminal failure
    """
    logger.info(f"🌊 API stream: '{request.query}'")
    return _sse_response(lambda: rag.ask_stream(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
        user_id=user.user_id,
    ))


# ==============================================================================
# MULTI-TURN CHAT
# ==============================================================================

@router.post("/chat/start", response_model=StartChatResponse)
async def start_chat(
    request: StartChatRequest,
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
):
    """Start a new conversation."""
    conv_id = rag.start_chat(request.title, user_id=user.user_id)
    return StartChatResponse(conversation_id=conv_id, title=request.title)


@router.post("/chat/{conversation_id}", response_model=ChatResponse)
async def chat(
    conversation_id: str,
    request: ChatRequest,
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
):
    """Send a message in a conversation."""
    logger.info(f"💬 API: Chat {conversation_id}: '{request.query}'")

    if not rag.owns_conversation(conversation_id, user.user_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        response = rag.chat(
            conversation_id,
            request.query,
            mode=request.mode,
            top_k=request.top_k,
            user_id=user.user_id,
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


@router.post("/chat/{conversation_id}/stream")
async def chat_stream(
    conversation_id: str,
    request: ChatRequest,
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
):
    """
    Send a message in a conversation, streaming the reply as Server-Sent Events.

    Same persistence as the non-streaming endpoint: the user message is saved
    before generation and the assistant message after, so history is identical.
    """
    logger.info(f"🌊 API stream chat {conversation_id}: '{request.query}'")

    if not rag.owns_conversation(conversation_id, user.user_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    return _sse_response(lambda: rag.chat_stream(
        conversation_id,
        request.query,
        top_k=request.top_k,
        user_id=user.user_id,
    ))


@router.get("/chat/{conversation_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    conversation_id: str,
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
):
    """Get all messages in a conversation."""
    if not rag.owns_conversation(conversation_id, user.user_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = rag.get_chat_history(conversation_id, user_id=user.user_id)
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
async def list_chats(
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
):
    """List the authenticated user's recent conversations."""
    convs = rag.list_chats(user_id=user.user_id)
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
async def list_documents(
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
):
    """List the authenticated user's ingested documents."""
    docs = rag.list_documents(user_id=user.user_id)
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs)
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
):
    """Delete one of the authenticated user's documents and its vectors."""
    deleted = rag.delete_document(document_id, user_id=user.user_id)
    if not deleted:
        # Also the response when the document belongs to someone else, so the
        # endpoint cannot be used to probe for other tenants' document ids.
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
async def get_stats(
    rag=Depends(get_rag),
    user: User = Depends(get_current_user),
):
    """Get query statistics."""
    status = rag.get_status()
    stats = status.get("query_stats", {})
    return StatsResponse(**stats) if stats else StatsResponse()
