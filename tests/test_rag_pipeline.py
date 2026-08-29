"""
Tests for DocuSenseRAG End-to-End Pipeline.

Phase 5: Complete RAG Pipeline tests.
All components are mocked for CI-safe testing.
"""

from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# ==============================================================================
# Mock objects
# ==============================================================================

@dataclass
class MockChunk:
    """Mock chunk from ingestion."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockPaperMetadata:
    """Mock paper metadata."""
    title: str = "BERT: Pre-training of Deep Bidirectional Transformers"
    authors: List[str] = field(default_factory=lambda: ["Jacob Devlin"])
    year: int = 2018
    venue: str = "NAACL"
    paper_type: str = "conference"
    confidence: float = 0.95

    def is_research_paper(self):
        return self.confidence > 0.5

    def to_dict(self):
        return {"title": self.title, "authors": self.authors, "year": self.year}


@dataclass
class MockPipelineResult:
    """Mock result from ingestion pipeline."""
    success: bool = True
    document_id: str = "doc_abc12345"
    filename: str = "bert.pdf"
    file_path: str = "papers/bert.pdf"
    total_chunks: int = 5
    chunks: List[MockChunk] = field(default_factory=list)
    paper_metadata: Optional[MockPaperMetadata] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None

    def __post_init__(self):
        if not self.chunks:
            self.chunks = [
                MockChunk(
                    chunk_id=f"chunk_{i}",
                    text=f"This is chunk {i} about BERT transformers.",
                    metadata={"chunk_index": i, "document_id": self.document_id}
                )
                for i in range(self.total_chunks)
            ]
        if self.paper_metadata is None:
            self.paper_metadata = MockPaperMetadata()


class MockPipelineResponse:
    """Mock generation pipeline response."""
    def __init__(self):
        self.answer = "BERT achieved 93.5% F1 on SST-2 (Devlin et al., 2018)."
        self.sources = [{"paper_title": "BERT Paper"}]
        self.papers_cited = ["BERT Paper"]
        self.reference_list = "[1] Devlin, J. (2018). BERT."
        self.bibtex = "@article{devlin2018bert}"
        self.confidence = 0.9
        self.has_citations = True
        self.num_sources = 3
        self.is_multi_paper = False
        self.total_time = 1.5


# ==============================================================================
# DocuSenseRAG Tests
# ==============================================================================

class TestDocuSenseRAG:
    """Tests for DocuSenseRAG."""

    def test_init_lazy(self):
        """Test lazy initialization — no components loaded at start."""
        from docusense.rag_pipeline import DocuSenseRAG
        rag = DocuSenseRAG()
        
        assert rag._ingestion_pipeline is None
        assert rag._embedding_generator is None
        assert rag._qdrant_store is None
        # Retrieval, generation, and conversation state are per-user caches,
        # empty until a scope is first used.
        assert rag._retrieval_pipelines == {}
        assert rag._generation_pipelines == {}
        assert rag._conversation_managers == {}

    def test_get_status(self):
        """Test status report shows uninitialized components."""
        from docusense.rag_pipeline import DocuSenseRAG
        rag = DocuSenseRAG()
        
        status = rag.get_status()
        assert status["components"]["ingestion"] is False
        assert status["components"]["embeddings"] is False
        assert status["components"]["qdrant"] is False

    @patch("docusense.rag_pipeline.DocuSenseRAG.ingestion_pipeline", new_callable=PropertyMock)
    @patch("docusense.rag_pipeline.DocuSenseRAG.embedding_generator", new_callable=PropertyMock)
    @patch("docusense.rag_pipeline.DocuSenseRAG.qdrant_store", new_callable=PropertyMock)
    def test_ingest_success(self, mock_qdrant, mock_embed, mock_ingest):
        """Test successful document ingestion."""
        from docusense.rag_pipeline import DocuSenseRAG
        import numpy as np

        # Setup mocks
        mock_pipeline = MagicMock()
        mock_pipeline.process_document.return_value = MockPipelineResult()
        mock_ingest.return_value = mock_pipeline

        mock_gen = MagicMock()
        mock_gen.embed_batch.return_value = np.random.rand(5, 384)
        mock_embed.return_value = mock_gen

        mock_store = MagicMock()
        mock_qdrant.return_value = mock_store

        rag = DocuSenseRAG()
        result = rag.ingest("papers/bert.pdf")

        assert result.success is True
        assert result.num_chunks == 5
        assert result.num_embeddings == 5
        assert result.is_research_paper is True
        assert "BERT" in result.paper_title

    @patch("docusense.rag_pipeline.DocuSenseRAG.ingestion_pipeline", new_callable=PropertyMock)
    def test_ingest_failure(self, mock_ingest):
        """Test ingestion failure handling."""
        from docusense.rag_pipeline import DocuSenseRAG

        mock_pipeline = MagicMock()
        mock_pipeline.process_document.return_value = MockPipelineResult(
            success=False,
            error="File not found"
        )
        mock_ingest.return_value = mock_pipeline

        rag = DocuSenseRAG()
        result = rag.ingest("nonexistent.pdf")

        assert result.success is False
        assert "File not found" in result.error

    @patch("docusense.rag_pipeline.DocuSenseRAG._generation_for")
    def test_ask(self, mock_gen):
        """Test asking a question."""
        from docusense.rag_pipeline import DocuSenseRAG

        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = MockPipelineResponse()
        mock_gen.return_value = mock_pipeline

        rag = DocuSenseRAG()
        response = rag.ask("What F1 score did BERT achieve?")

        assert "93.5%" in response.answer
        mock_pipeline.generate.assert_called_once()

    @patch("docusense.rag_pipeline.DocuSenseRAG._generation_for")
    def test_compare(self, mock_gen):
        """Test compare mode."""
        from docusense.rag_pipeline import DocuSenseRAG

        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = MockPipelineResponse()
        mock_gen.return_value = mock_pipeline

        rag = DocuSenseRAG()
        rag.compare("Compare BERT and GPT-2")

        call_args = mock_pipeline.generate.call_args
        assert call_args.kwargs.get("mode") == "compare"
        assert call_args.kwargs.get("top_k") == 10

    @patch("docusense.rag_pipeline.DocuSenseRAG._conversations_for")
    def test_start_chat(self, mock_conv):
        """Test starting a chat."""
        from docusense.rag_pipeline import DocuSenseRAG

        mock_mgr = MagicMock()
        mock_mgr.start_conversation.return_value = "conv_abc123"
        mock_conv.return_value = mock_mgr

        rag = DocuSenseRAG()
        conv_id = rag.start_chat("BERT Chat")

        assert conv_id == "conv_abc123"
        mock_mgr.start_conversation.assert_called_once_with("BERT Chat", user_id=None)

    @patch("docusense.rag_pipeline.DocuSenseRAG._conversations_for")
    def test_chat(self, mock_conv):
        """Test chat with conversation."""
        from docusense.rag_pipeline import DocuSenseRAG

        mock_mgr = MagicMock()
        mock_conv.return_value = mock_mgr

        rag = DocuSenseRAG()
        rag.chat("conv_abc123", "What is BERT?")

        mock_mgr.chat.assert_called_once_with(
            "conv_abc123", "What is BERT?", mode="answer", top_k=5, filters=None
        )

    @patch("docusense.rag_pipeline.DocuSenseRAG.ingestion_pipeline", new_callable=PropertyMock)
    def test_list_documents(self, mock_ingest):
        """Test listing documents."""
        from docusense.rag_pipeline import DocuSenseRAG
        
        mock_pipeline = MagicMock()
        mock_doc = MagicMock()
        mock_doc.document_id = "doc_123"
        mock_doc.filename = "bert.pdf"
        mock_doc.file_type = "pdf"
        mock_doc.total_chunks = 10
        mock_doc.processing_date = "2026-03-08"
        mock_doc.metadata = {"is_research_paper": True, "paper_metadata": {"title": "BERT"}}
        mock_pipeline.storage.get_all_documents.return_value = [mock_doc]
        mock_ingest.return_value = mock_pipeline

        rag = DocuSenseRAG()
        docs = rag.list_documents()

        assert len(docs) == 1
        assert docs[0]["filename"] == "bert.pdf"
        assert docs[0]["is_research_paper"] is True


# ==============================================================================
# IngestResult Tests
# ==============================================================================

class TestIngestResult:
    """Tests for IngestResult dataclass."""

    def test_str_success(self):
        """Test success string representation."""
        from docusense.rag_pipeline import IngestResult
        
        result = IngestResult(
            success=True,
            document_id="doc_123",
            filename="bert.pdf",
            num_chunks=10,
            num_embeddings=10,
            is_research_paper=True,
            paper_title="BERT Paper",
            processing_time=5.2
        )
        
        output = str(result)
        assert "✅" in output
        assert "bert.pdf" in output
        assert "10 chunks" in output

    def test_str_failure(self):
        """Test failure string representation."""
        from docusense.rag_pipeline import IngestResult
        
        result = IngestResult(
            success=False,
            document_id="doc_123",
            filename="bad.pdf",
            error="File not found"
        )
        
        output = str(result)
        assert "❌" in output
        assert "File not found" in output
