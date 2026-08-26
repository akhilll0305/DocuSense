"""
FastAPI Application - DocuSense API Server.

Phase 7: API & Web UI

Run with: uvicorn docusense.api.app:app --reload

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from loguru import logger

from docusense.config.settings import settings
from docusense.api.auth_routes import router as auth_router
from docusense.api.deps import set_rag_instance, set_user_store
from docusense.api.routes import router


# Path to web UI static files
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG system and user store on startup, clean up on shutdown."""
    logger.info("🚀 Starting DocuSense API...")

    from docusense.auth import UserStore
    from docusense.rag_pipeline import DocuSenseRAG

    user_store = UserStore()
    set_user_store(user_store)

    rag = DocuSenseRAG()
    set_rag_instance(rag)

    logger.success("✅ DocuSense API ready")
    logger.info(f"🌐 Web UI: http://localhost:8000")
    yield

    logger.info("🛑 Shutting down DocuSense API...")
    rag.close()
    user_store.close()


app = FastAPI(
    title="DocuSense",
    description=(
        "Research Paper Analysis RAG System — "
        "Ingest papers, ask questions, get answers with citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS. The web UI is served from this same origin, so cross-origin access is
# only needed for external clients and is opt-in via CORS_ALLOW_ORIGINS.
# "*" with allow_credentials is rejected by browsers anyway, so credentials are
# only enabled for an explicit origin list.
_origins = settings.cors_allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(auth_router)
app.include_router(router)

# Serve static web UI files
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    logger.info(f"📁 Serving web UI from {WEB_DIR}")


@app.get("/")
async def root():
    """Redirect to the landing page."""
    return RedirectResponse(url="/static/index.html")
