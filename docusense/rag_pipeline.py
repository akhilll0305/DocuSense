"""
DocuSense RAG Pipeline - End-to-end Research Paper Analysis.

Phase 5: Complete RAG Pipeline (Final Step)

PURPOSE:
--------
Single entry point for the entire DocuSense system:
  PDF → Ingest → Embed → Store in Qdrant → Retrieve → Answer with Citations

USAGE:
------
```python
from docusense.rag_pipeline import DocuSenseRAG

rag = DocuSenseRAG()

# 1. Ingest a paper
rag.ingest("papers/bert.pdf")

# 2. Ask questions
response = rag.ask("What F1 score did BERT achieve on SST-2?")
print(response.answer)
# "BERT achieved 93.5% F1 on SST-2 (Devlin et al., 2018, Results)"

# 3. Multi-turn chat
conv_id = rag.start_chat("BERT Analysis")
r1 = rag.chat(conv_id, "What is BERT?")
r2 = rag.chat(conv_id, "What about its accuracy?")  # uses context

# 4. Compare papers
response = rag.compare("How do BERT and GPT-2 differ?")
```

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time

from loguru import logger


import uuid

try:
    from qdrant_client.models import PointStruct
except ImportError:
    PointStruct = None


@dataclass
class IngestResult:
    """Result of document ingestion into the RAG system."""
    success: bool
    document_id: str
    filename: str
    num_chunks: int = 0
    num_embeddings: int = 0
    is_research_paper: bool = False
    paper_title: Optional[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            paper = f" [{self.paper_title}]" if self.paper_title else ""
            return (
                f"✅ Ingested {self.filename}: "
                f"{self.num_chunks} chunks, {self.num_embeddings} embeddings "
                f"({self.processing_time:.1f}s){paper}"
            )
        return f"❌ Failed to ingest {self.filename}: {self.error}"


class DocuSenseRAG:
    """
    End-to-end Research Paper Analysis RAG System.

    Wires together all DocuSense components:
    - Ingestion Pipeline (Phase 1): PDF → chunks
    - Embedding Generator (Phase 2): chunks → vectors
    - Qdrant Store (Phase 2): vector storage + search
    - Retrieval Pipeline (Phase 3): query → relevant chunks
    - Generation Pipeline (Phase 4): chunks → answer with citations
    - Conversation Manager (Phase 5): multi-turn dialog

    All components are lazily initialized to avoid loading heavy models
    until actually needed.
    """

    def __init__(
        self,
        auto_init: bool = False,
        enable_images: bool = False,
        enable_paper_extraction: bool = True
    ):
        """
        Initialize DocuSenseRAG.

        Args:
            auto_init: If True, initialize all components immediately.
                       If False (default), components are lazily loaded.
            enable_images: Process images in documents (default False for speed)
            enable_paper_extraction: Extract paper metadata (default True)
        """
        self.enable_images = enable_images
        self.enable_paper_extraction = enable_paper_extraction

        # Component placeholders (lazy init)
        self._ingestion_pipeline = None
        self._embedding_generator = None
        self._qdrant_store = None
        self._conversation_managers: Dict[str, Any] = {}

        # Retrieval and generation are per-user: BM25 holds an in-memory
        # corpus that must not be shared across tenants.
        self._retrieval_pipelines: Dict[str, Any] = {}
        self._generation_pipelines: Dict[str, Any] = {}

        logger.info("📚 DocuSenseRAG created")

        if auto_init:
            self._init_all()

    # ==================================================================
    # LAZY COMPONENT INITIALIZATION
    # ==================================================================

    @property
    def ingestion_pipeline(self):
        """Lazily initialize ingestion pipeline."""
        if self._ingestion_pipeline is None:
            from docusense.ingestion.pipeline import DocumentPipeline
            self._ingestion_pipeline = DocumentPipeline(
                enable_images=self.enable_images,
                enable_paper_extraction=self.enable_paper_extraction
            )
            logger.info("📄 Ingestion pipeline initialized")
        return self._ingestion_pipeline

    @property
    def embedding_generator(self):
        """Lazily initialize embedding generator."""
        if self._embedding_generator is None:
            from docusense.embeddings.embedding_generator import EmbeddingGenerator
            self._embedding_generator = EmbeddingGenerator()
            logger.info("🧮 Embedding generator initialized")
        return self._embedding_generator

    @property
    def qdrant_store(self):
        """Lazily initialize Qdrant vector store."""
        if self._qdrant_store is None:
            from docusense.vectorstore.qdrant_store import QdrantVectorStore
            self._qdrant_store = QdrantVectorStore()
            logger.info("🗄️ Qdrant store initialized")
        return self._qdrant_store

    def _load_bm25_corpus(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load stored chunks as dicts for BM25 indexing.

        BM25 is an in-memory index built from full chunk text, so it has to be
        rebuilt from SQLite on startup and refreshed whenever documents change.

        Args:
            user_id: Scope the corpus to one owner. Required for multi-tenant
                use — a shared corpus would let BM25 surface another user's text.
        """
        try:
            records = self.ingestion_pipeline.storage.get_all_chunks(user_id=user_id)
        except Exception as e:
            logger.warning(f"Could not load chunks for BM25: {e}")
            return []

        corpus = []
        for r in records:
            meta = dict(r.metadata or {})
            meta.setdefault("document_id", r.document_id)
            meta.setdefault("chunk_index", r.chunk_index)
            meta.setdefault("header_path", r.header_path)
            meta.setdefault("page_number", r.page_number)
            corpus.append({
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                # Nested copy: BM25-sourced hits read chunk['metadata'], and
                # without it citations lose paper title/authors/year.
                "metadata": meta,
                **meta,
            })
        return corpus

    # Sentinel key for single-tenant use (CLI scripts, tests) where there is
    # no authenticated user.
    _GLOBAL_SCOPE = "__global__"

    def _scope_key(self, user_id: Optional[str]) -> str:
        return user_id or self._GLOBAL_SCOPE

    def _retrieval_for(self, user_id: Optional[str] = None):
        """
        Get the retrieval pipeline for one user, building it on first use.

        BM25 keeps an in-memory corpus, so it cannot be shared across tenants:
        each user gets their own index built from only their documents. The
        vector store is shared, and queries against it are filtered by user_id.
        """
        key = self._scope_key(user_id)
        if key not in self._retrieval_pipelines:
            from docusense.retrieval.retrieval_pipeline import RetrievalPipeline

            from docusense.config.settings import settings

            corpus = self._load_bm25_corpus(user_id)
            # mode="accurate" is the only mode that respects the explicit
            # flags; "balanced" (the previous default here) forced reranking
            # off, so `USE_RERANKING` in .env did nothing at all. Measured on
            # QASPER, that silently cost 36% MRR: 0.2041 with the reranker off
            # against 0.2777 with it on, over 259 questions.
            self._retrieval_pipelines[key] = RetrievalPipeline(
                vector_store=self.qdrant_store,
                chunks=corpus,
                enable_query_processing=True,
                enable_hybrid_search=True,
                enable_reranking=settings.use_reranking,
                mode="accurate",
            )
            logger.info(
                f"🔍 Retrieval pipeline ready for {key} "
                f"({len(corpus)} chunks indexed for BM25)"
            )
        return self._retrieval_pipelines[key]

    def _generation_for(self, user_id: Optional[str] = None):
        """Get the generation pipeline bound to this user's retrieval scope."""
        key = self._scope_key(user_id)
        if key not in self._generation_pipelines:
            from docusense.generation.generation_pipeline import GenerationPipeline

            self._generation_pipelines[key] = GenerationPipeline(
                retrieval_pipeline=self._retrieval_for(user_id)
            )
        return self._generation_pipelines[key]

    @property
    def retrieval_pipeline(self):
        """Retrieval pipeline for the unscoped/global corpus."""
        return self._retrieval_for(None)

    @property
    def generation_pipeline(self):
        """Generation pipeline for the unscoped/global corpus."""
        return self._generation_for(None)

    def refresh_retrieval_index(self, user_id: Optional[str] = None) -> None:
        """
        Rebuild a user's BM25 index from storage.

        Vector search picks up new documents immediately because Qdrant is
        queried live, but BM25 holds an in-memory corpus that goes stale after
        ingestion or deletion.
        """
        key = self._scope_key(user_id)
        pipeline = self._retrieval_pipelines.get(key)
        if pipeline is None:
            return  # Nothing built yet; it will load fresh on first use.

        hybrid = getattr(pipeline, "hybrid_search", None)
        if hybrid is None:
            return

        corpus = self._load_bm25_corpus(user_id)
        pipeline.chunks = corpus
        if corpus:
            hybrid.index_chunks(corpus)
            logger.info(f"🔄 BM25 index refreshed for {key} ({len(corpus)} chunks)")

    @staticmethod
    def _scoped_filters(
        filters: Optional[Dict[str, Any]], user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Add the tenant constraint to a filter set.

        The vector store is shared across users, so every search must carry
        user_id or one tenant's query would match another's chunks.
        """
        if user_id is None:
            return filters
        scoped = dict(filters or {})
        scoped["user_id"] = user_id
        return scoped

    @property
    def conversation_manager(self):
        """Conversation manager for the unscoped/global corpus."""
        return self._conversations_for(None)

    def _init_all(self):
        """Force-initialize all components."""
        logger.info("🚀 Initializing all DocuSenseRAG components...")
        _ = self.ingestion_pipeline
        _ = self.embedding_generator
        _ = self.qdrant_store
        _ = self.retrieval_pipeline
        _ = self.generation_pipeline
        _ = self.conversation_manager
        logger.success("✅ All components initialized")

    # ==================================================================
    # DOCUMENT INGESTION
    # ==================================================================

    def ingest(
        self,
        file_path: str | Path,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        original_filename: Optional[str] = None
    ) -> IngestResult:
        """
        Ingest a document: convert → chunk → embed → store in Qdrant.

        Args:
            file_path: Path to PDF, DOCX, or TXT file
            document_id: Optional custom document ID
            metadata: Optional metadata dict
            user_id: Owning user. Stored on the document row and stamped onto
                every vector payload so retrieval can filter by tenant.
            original_filename: Name to record for the document. Uploads arrive
                as a temp file, so without this the stored name is the
                temp basename and the document list shows `tmpXXXXXXXX.pdf`.

        Returns:
            IngestResult with success status and details
        """
        file_path = Path(file_path)
        start_time = time.time()

        display_name = original_filename or file_path.name

        logger.info(f"📥 Ingesting: {display_name}")

        # Step 1: Run ingestion pipeline (convert → chunk → store in SQLite)
        try:
            pipeline_result = self.ingestion_pipeline.process_document(
                file_path, document_id, metadata, user_id=user_id,
                original_filename=display_name
            )
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            return IngestResult(
                success=False,
                document_id=document_id or "unknown",
                filename=display_name,
                error=str(e),
                processing_time=time.time() - start_time
            )

        if not pipeline_result.success:
            return IngestResult(
                success=False,
                document_id=pipeline_result.document_id,
                filename=display_name,
                error=pipeline_result.error,
                processing_time=time.time() - start_time
            )

        # Step 2: Generate embeddings for chunks
        chunks = pipeline_result.chunks
        if not chunks:
            return IngestResult(
                success=True,
                document_id=pipeline_result.document_id,
                filename=display_name,
                num_chunks=0,
                processing_time=time.time() - start_time
            )

        texts = [chunk.text for chunk in chunks]
        logger.info(f"🧮 Generating embeddings for {len(texts)} chunks...")

        try:
            embeddings = self.embedding_generator.embed_batch(
                texts, show_progress=True
            )
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return IngestResult(
                success=False,
                document_id=pipeline_result.document_id,
                filename=display_name,
                num_chunks=len(chunks),
                error=f"Embedding failed: {e}",
                processing_time=time.time() - start_time
            )

        # Step 3: Store in Qdrant with metadata
        logger.info(f"🗄️ Storing {len(embeddings)} vectors in Qdrant...")

        try:
            # Ensure collection exists
            self.qdrant_store.create_collection()

            # Build all points
            points = []
            for i, chunk in enumerate(chunks):
                payload = {
                    "text": chunk.text,
                    "document_id": chunk.metadata.get("document_id", pipeline_result.document_id),
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.metadata.get("chunk_index", i),
                    "has_code": chunk.metadata.get("has_code", False),
                    "has_tables": chunk.metadata.get("has_tables", False),
                    # Tenant key: every search filters on this.
                    "user_id": user_id or "",
                }

                # Add paper metadata if available
                if pipeline_result.paper_metadata and pipeline_result.paper_metadata.is_research_paper():
                    pm = pipeline_result.paper_metadata
                    payload.update({
                        "paper_title": pm.title or "",
                        "authors": pm.authors or [],
                        "venue": pm.venue or "",
                        "section_type": chunk.metadata.get("section_type", ""),
                        "paper_type": pm.paper_type or "",
                        "has_equations": chunk.metadata.get("has_equations", False),
                        "has_citations": chunk.metadata.get("has_citations", False),
                    })
                    # An unknown year is left off the payload rather than
                    # written as 0. A 0 matches no useful filter and renders
                    # as "(Smith, 0)"; an absent key renders as "n.d.".
                    if pm.year:
                        payload["year"] = pm.year

                if PointStruct is not None:
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embeddings[i].tolist(),
                        payload=payload
                    ))
                else:
                    # Fallback for environments without qdrant_client
                    # (should only happen in mocked tests)
                    points.append({
                        "id": str(uuid.uuid4()),
                        "vector": embeddings[i].tolist(),
                        "payload": payload
                    })

            # Batch upsert
            self.qdrant_store.client.upsert(
                collection_name=self.qdrant_store.collection_name,
                points=points
            )

            logger.success(f"✅ Stored {len(embeddings)} vectors in Qdrant")
        except Exception as e:
            logger.error(f"❌ Qdrant storage failed: {e}")
            return IngestResult(
                success=False,
                document_id=pipeline_result.document_id,
                filename=display_name,
                num_chunks=len(chunks),
                num_embeddings=len(embeddings),
                error=f"Qdrant storage failed: {e}",
                processing_time=time.time() - start_time
            )

        elapsed = time.time() - start_time
        paper_title = None
        is_paper = False
        if pipeline_result.paper_metadata and pipeline_result.paper_metadata.is_research_paper():
            paper_title = pipeline_result.paper_metadata.title
            is_paper = True

        result = IngestResult(
            success=True,
            document_id=pipeline_result.document_id,
            filename=display_name,
            num_chunks=len(chunks),
            num_embeddings=len(embeddings),
            is_research_paper=is_paper,
            paper_title=paper_title,
            processing_time=elapsed
        )

        # New chunks are live in Qdrant, but BM25 holds a stale in-memory copy.
        self.refresh_retrieval_index(user_id)

        logger.success(f"🎉 {result}")
        return result

    def ingest_batch(
        self,
        file_paths: List[str | Path],
        skip_errors: bool = True
    ) -> List[IngestResult]:
        """
        Ingest multiple documents.

        Args:
            file_paths: List of file paths
            skip_errors: Continue on failure (default True)

        Returns:
            List of IngestResults
        """
        logger.info(f"📚 Batch ingesting {len(file_paths)} documents...")
        results = []

        for fp in file_paths:
            result = self.ingest(fp)
            results.append(result)
            if not result.success and not skip_errors:
                break

        successes = sum(1 for r in results if r.success)
        logger.info(f"📚 Batch complete: {successes}/{len(results)} successful")
        return results

    # ==================================================================
    # QUESTION ANSWERING
    # ==================================================================

    def ask(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "answer",
        user_id: Optional[str] = None
    ):
        """
        Ask a question and get an answer with citations.

        Args:
            query: Natural language question
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters
            mode: "answer" (default), "compare", "conflicts"
            user_id: Restrict retrieval to this user's documents

        Returns:
            PipelineResponse with answer, citations, references
        """
        logger.info(f"❓ Asking: '{query}'")
        return self._generation_for(user_id).generate(
            query=query,
            top_k=top_k,
            filters=self._scoped_filters(filters, user_id),
            mode=mode
        )

    def ask_stream(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ):
        """
        Ask a question, yielding the answer as it is generated.

        Args:
            query: Natural language question
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters
            user_id: Restrict retrieval to this user's documents

        Yields:
            (kind, payload) pairs — "status", "token", "done", or "error".
            See GenerationPipeline.generate_stream.
        """
        logger.info(f"🌊 Streaming answer for: '{query}'")
        return self._generation_for(user_id).generate_stream(
            query=query,
            top_k=top_k,
            filters=self._scoped_filters(filters, user_id),
        )

    def compare(self, query: str, top_k: int = 10, user_id: Optional[str] = None):
        """Compare findings across multiple papers."""
        return self.ask(query, top_k=top_k, mode="compare", user_id=user_id)

    # ==================================================================
    # MULTI-TURN CHAT
    # ==================================================================

    def _conversations_for(self, user_id: Optional[str] = None):
        """
        Get the conversation manager bound to this user's generation scope.

        Each manager wraps a generation pipeline, which is per-user, so
        managers are cached per user as well. They share one ConversationStore
        because that state lives in SQLite and is scoped by query.
        """
        key = self._scope_key(user_id)
        if key not in self._conversation_managers:
            from docusense.generation.conversation_manager import ConversationManager

            self._conversation_managers[key] = ConversationManager(
                generation_pipeline=self._generation_for(user_id)
            )
        return self._conversation_managers[key]

    def owns_conversation(self, conversation_id: str, user_id: Optional[str]) -> bool:
        """
        Whether a user may access a conversation.

        Unscoped callers (CLI, tests) pass user_id=None and are unrestricted.
        """
        if user_id is None:
            return True
        return self._conversations_for(user_id).get_owner(conversation_id) == user_id

    def start_chat(self, title: str = "New Chat", user_id: Optional[str] = None) -> str:
        """
        Start a new chat conversation.

        Returns:
            conversation_id
        """
        return self._conversations_for(user_id).start_conversation(title, user_id=user_id)

    def chat(
        self,
        conversation_id: str,
        query: str,
        mode: str = "answer",
        top_k: int = 5,
        user_id: Optional[str] = None
    ):
        """
        Chat with conversation context.

        Args:
            conversation_id: Active conversation
            query: User's question
            mode: "answer", "compare", or "conflicts"
            top_k: Number of chunks to retrieve
            user_id: Restrict retrieval to this user's documents

        Returns:
            ChatResponse with answer and metadata
        """
        return self._conversations_for(user_id).chat(
            conversation_id,
            query,
            mode=mode,
            top_k=top_k,
            filters=self._scoped_filters(None, user_id),
        )

    def chat_stream(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None
    ):
        """
        Chat with conversation context, yielding the answer as it is generated.

        Yields:
            (kind, payload) pairs — "status", "token", "done", or "error".
        """
        return self._conversations_for(user_id).chat_stream(
            conversation_id,
            query,
            top_k=top_k,
            filters=self._scoped_filters(None, user_id),
        )

    def get_chat_history(self, conversation_id: str, user_id: Optional[str] = None):
        """Get all messages in a conversation."""
        return self._conversations_for(user_id).get_history(conversation_id)

    def list_chats(self, user_id: Optional[str] = None):
        """List recent conversations for a user."""
        return self._conversations_for(user_id).list_conversations(user_id=user_id)

    # ==================================================================
    # DOCUMENT MANAGEMENT
    # ==================================================================

    def list_documents(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List ingested documents, scoped to a user when given."""
        try:
            storage = self.ingestion_pipeline.storage
            docs = storage.get_all_documents(user_id=user_id)
            return [
                {
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "total_chunks": doc.total_chunks,
                    "processing_date": doc.processing_date,
                    "is_research_paper": doc.metadata.get("is_research_paper", False),
                    "paper_title": doc.metadata.get("paper_metadata", {}).get("title", "")
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            return []

    def owns_document(self, document_id: str, user_id: Optional[str]) -> bool:
        """
        Whether a user may act on a document.

        Unscoped callers (CLI, tests) pass user_id=None and are unrestricted.
        """
        if user_id is None:
            return True
        owner = self.ingestion_pipeline.storage.get_document_owner(document_id)
        return owner == user_id

    def delete_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a document, its chunks, and its vectors.

        Removes the Qdrant points first: if that fails we still have the SQLite
        rows, so the document remains listed and the delete can be retried. The
        reverse order would strand unreachable vectors with no record of them.

        Returns False when the document is not owned by user_id, so a caller
        cannot delete another tenant's data.
        """
        if not self.owns_document(document_id, user_id):
            logger.warning(f"Refusing delete of {document_id}: not owned by {user_id}")
            return False

        try:
            self.qdrant_store.delete_by_document(document_id)
        except Exception as e:
            logger.error(f"❌ Failed to delete vectors for {document_id}: {e}")
            return False

        try:
            deleted = self.ingestion_pipeline.storage.delete_document(document_id)
        except Exception as e:
            logger.error(f"❌ Failed to delete document record: {e}")
            return False

        if deleted:
            self.refresh_retrieval_index(user_id)
            logger.info(f"🗑️ Deleted document and vectors: {document_id}")
        return deleted

    # ==================================================================
    # SYSTEM INFO
    # ==================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        status = {
            "components": {
                "ingestion": self._ingestion_pipeline is not None,
                "embeddings": self._embedding_generator is not None,
                "qdrant": self._qdrant_store is not None,
                "retrieval": bool(self._retrieval_pipelines),
                "generation": bool(self._generation_pipelines),
                "conversation": bool(self._conversation_managers),
            }
        }

        # Query stats are shared state in SQLite, so any live manager can report them.
        for manager in self._conversation_managers.values():
            status["query_stats"] = manager.get_query_stats()
            break

        return status

    def close(self):
        """Close all components."""
        if self._ingestion_pipeline:
            self._ingestion_pipeline.close()
        for manager in self._conversation_managers.values():
            manager.close()
        logger.info("📚 DocuSenseRAG closed")
