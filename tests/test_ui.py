"""
Tests for Gradio UI handlers.

Phase 7: UI tests — handler logic is tested with mocked RAG.
Gradio rendering is not tested (requires browser).
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# ==============================================================================
# Mock objects
# ==============================================================================

@dataclass
class MockIngestResult:
    success: bool = True
    document_id: str = "doc_test"
    filename: str = "bert.pdf"
    num_chunks: int = 10
    num_embeddings: int = 10
    is_research_paper: bool = True
    paper_title: str = "BERT Paper"
    processing_time: float = 2.5
    error: Optional[str] = None


@dataclass
class MockPipelineResponse:
    answer: str = "BERT achieved 93.5% (Devlin et al., 2018)."
    sources: List[Dict[str, Any]] = field(default_factory=list)
    papers_cited: List[str] = field(default_factory=lambda: ["BERT"])
    reference_list: str = "[1] Devlin (2018)"
    confidence: float = 0.9
    has_citations: bool = True
    num_sources: int = 1
    total_time: float = 1.5


@dataclass
class MockChatResponse:
    answer: str = "BERT is a model."
    conversation_id: str = "conv_test"
    message_id: str = "msg_test"
    sources: List[Dict[str, Any]] = field(default_factory=list)
    papers_cited: List[str] = field(default_factory=list)
    reference_list: str = ""
    confidence: float = 0.8
    response_time: float = 1.0
    turn_number: int = 1


# ==============================================================================
# Handler Tests (no Gradio rendering needed)
# ==============================================================================

class TestDocuSenseUIHandlers:
    """Test UI handler methods with mocked RAG."""

    @pytest.fixture
    def ui(self):
        """Create UI with mocked RAG."""
        pytest.importorskip("gradio")
        from docusense.ui.gradio_app import DocuSenseUI

        mock_rag = MagicMock()
        mock_rag.ingest.return_value = MockIngestResult()
        mock_rag.ask.return_value = MockPipelineResponse()
        mock_rag.start_chat.return_value = "conv_test"
        mock_rag.chat.return_value = MockChatResponse()
        mock_rag.list_documents.return_value = [
            {"filename": "bert.pdf", "total_chunks": 10,
             "is_research_paper": True, "paper_title": "BERT"}
        ]

        ui_instance = DocuSenseUI(rag=mock_rag)
        return ui_instance

    def test_handle_upload_success(self, ui):
        """Test successful upload."""
        mock_file = MagicMock()
        mock_file.name = "bert.pdf"
        result = ui.handle_upload(mock_file)
        assert "✅" in result
        assert "10" in result  # num_chunks

    def test_handle_upload_none(self, ui):
        """Test upload with no file."""
        result = ui.handle_upload(None)
        assert "No file" in result

    def test_handle_ask(self, ui):
        """Test asking a question."""
        result = ui.handle_ask("What is BERT?", "Answer", 5)
        assert "93.5%" in result
        assert "References" in result

    def test_handle_ask_empty(self, ui):
        """Test asking empty question."""
        result = ui.handle_ask("", "Answer", 5)
        assert "Please enter" in result

    def test_handle_chat(self, ui):
        """Test chat message."""
        msg, history = ui.handle_chat("What is BERT?", [], "Answer")
        assert msg == ""  # Input cleared
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_handle_chat_empty(self, ui):
        """Test empty chat message."""
        msg, history = ui.handle_chat("", [], "Answer")
        assert history == []

    def test_handle_new_chat(self, ui):
        """Test starting new chat."""
        msg, history = ui.handle_new_chat()
        assert msg == ""
        assert history == []
        assert ui._active_conversation is None

    def test_handle_list_documents(self, ui):
        """Test listing documents."""
        result = ui.handle_list_documents()
        assert "bert.pdf" in result
        assert "BERT" in result

    def test_handle_list_documents_empty(self, ui):
        """Test listing when no documents."""
        ui.rag.list_documents.return_value = []
        result = ui.handle_list_documents()
        assert "No documents" in result

    def test_handle_run_benchmark(self, ui):
        """Test running benchmark."""
        result = ui.handle_run_benchmark()
        assert "Benchmark" in result

    def test_build_returns_blocks(self, ui):
        """Test that build() returns a Gradio Blocks object."""
        import gradio as gr
        demo = ui.build()
        assert isinstance(demo, gr.Blocks)
