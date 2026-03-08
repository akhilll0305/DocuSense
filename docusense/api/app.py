"""
FastAPI Application - DocuSense API Server.

Phase 7: API & UI (Step 3)

Run with: uvicorn docusense.api.app:app --reload

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from docusense.api.routes import router, set_rag_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG system on startup, cleanup on shutdown."""
    logger.info("🚀 Starting DocuSense API...")

    from docusense.rag_pipeline import DocuSenseRAG
    rag = DocuSenseRAG()
    set_rag_instance(rag)

    logger.success("✅ DocuSense API ready")
    yield

    logger.info("🛑 Shutting down DocuSense API...")
    rag.close()


app = FastAPI(
    title="DocuSense",
    description=(
        "Research Paper Analysis RAG System — "
        "Ingest papers, ask questions, get answers with citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Gradio and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "DocuSense",
        "description": "Research Paper Analysis RAG System",
        "docs": "/docs",
        "version": "1.0.0",
    }
