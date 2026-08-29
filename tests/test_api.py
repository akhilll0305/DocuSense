"""
Tests for FastAPI API endpoints.

Phase 7: API tests using TestClient with mocked RAG backend.
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import io


# ==============================================================================
# Mock objects
# ==============================================================================

@dataclass
class MockIngestResult:
    success: bool = True
    document_id: str = "doc_test123"
    filename: str = "test.pdf"
    num_chunks: int = 10
    num_embeddings: int = 10
    is_research_paper: bool = True
    paper_title: str = "Test Paper"
    processing_time: float = 2.5
    error: Optional[str] = None


@dataclass
class MockPipelineResponse:
    answer: str = "BERT achieved 93.5% F1 (Devlin et al., 2018)."
    sources: List[Dict[str, Any]] = field(default_factory=lambda: [{"paper_title": "BERT"}])
    papers_cited: List[str] = field(default_factory=lambda: ["BERT Paper"])
    reference_list: str = "[1] Devlin (2018)"
    confidence: float = 0.9
    has_citations: bool = True
    num_sources: int = 1
    total_time: float = 1.5
    is_multi_paper: bool = False
    bibtex: str = ""


@dataclass
class MockChatResponse:
    answer: str = "BERT is a transformer model."
    conversation_id: str = "conv_test123"
    message_id: str = "msg_test456"
    sources: List[Dict[str, Any]] = field(default_factory=list)
    papers_cited: List[str] = field(default_factory=list)
    reference_list: str = ""
    confidence: float = 0.8
    response_time: float = 1.0
    turn_number: int = 1


@dataclass
class MockMessage:
    role: str = "user"
    content: str = "Test message"
    timestamp: str = "2026-03-08T00:00:00"
    sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MockConversation:
    conversation_id: str = "conv_abc"
    title: str = "Test Chat"
    created_at: str = "2026-03-08T00:00:00"
    updated_at: str = "2026-03-08T00:00:00"


# ==============================================================================
# Test fixtures
# ==============================================================================

@pytest.fixture
def mock_rag():
    """Create a mocked RAG instance."""
    rag = MagicMock()
    rag.ingest.return_value = MockIngestResult()
    rag.ask.return_value = MockPipelineResponse()
    rag.start_chat.return_value = "conv_test123"
    rag.chat.return_value = MockChatResponse()
    rag.get_chat_history.return_value = [
        MockMessage(role="user", content="Hello"),
        MockMessage(role="assistant", content="Hi!"),
    ]
    rag.list_chats.return_value = [MockConversation()]
    rag.list_documents.return_value = [
        {
            "document_id": "doc_1",
            "filename": "bert.pdf",
            "file_type": "pdf",
            "total_chunks": 10,
            "processing_date": "2026-03-08",
            "is_research_paper": True,
            "paper_title": "BERT"
        }
    ]
    rag.delete_document.return_value = True
    rag.get_status.return_value = {
        "components": {"ingestion": True, "embeddings": True},
        "query_stats": {"total_queries": 5, "avg_response_time": 1.2}
    }
    rag.close.return_value = None
    return rag


@pytest.fixture
def client(mock_rag):
    """
    Test client with a mocked RAG backend and a stubbed authenticated user.

    These tests cover endpoint wiring, not authentication; get_current_user is
    overridden so they exercise the handlers directly. Real credential handling
    and tenant isolation are covered in tests/test_auth.py.
    """
    from fastapi.testclient import TestClient
    from docusense.api.routes import router
    from docusense.api.deps import get_rag, get_current_user
    from docusense.auth import User

    # Build a minimal app that doesn't run the real lifespan
    from fastapi import FastAPI
    test_app = FastAPI()
    test_app.include_router(router)

    test_user = User(
        user_id="usr_test",
        email="test@example.com",
        name="Test",
        password_hash="",
        created_at="2026-01-01T00:00:00",
    )

    test_app.dependency_overrides[get_rag] = lambda: mock_rag
    test_app.dependency_overrides[get_current_user] = lambda: test_user

    with TestClient(test_app) as c:
        yield c



# ==============================================================================
# API Tests
# ==============================================================================

class TestRootEndpoint:
    def test_root(self, client):
        """Test that API is accessible."""
        response = client.get("/api/health")
        assert response.status_code == 200


class TestIngestEndpoint:
    def test_ingest_file(self, client):
        """Test file ingestion."""
        file_content = b"Test PDF content"
        response = client.post(
            "/api/ingest",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["num_chunks"] == 10

    def test_ingest_passes_the_uploaded_name_not_the_temp_name(self, client, mock_rag):
        """
        The upload is written to a NamedTemporaryFile, so the path handed to
        rag.ingest() has a throwaway basename. The name the user uploaded has
        to travel separately or it is what gets persisted, and the document
        list reads "tmp185or2yx.pdf".
        """
        client.post(
            "/api/ingest",
            files={"file": ("Attention Is All You Need.pdf",
                            io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
        )
        args, kwargs = mock_rag.ingest.call_args
        assert kwargs["original_filename"] == "Attention Is All You Need.pdf"
        # The positional path really is a temp file, so the kwarg is load-bearing.
        temp_path = args[0]
        assert "Attention" not in str(temp_path)

    def test_ingest_strips_directories_from_the_uploaded_name(self, client, mock_rag):
        """A client-supplied filename is untrusted; only the basename is kept."""
        client.post(
            "/api/ingest",
            files={"file": ("../../etc/passwd.pdf",
                            io.BytesIO(b"x"), "application/pdf")},
        )
        assert mock_rag.ingest.call_args.kwargs["original_filename"] == "passwd.pdf"


class TestIngestLimits:
    """
    A public instance has an open sign-up form; without these an upload
    endpoint is a public disk.
    """

    def test_no_limit_by_default(self, client, mock_rag):
        """Locally it is the user's own machine and their own disk."""
        from docusense.config.settings import settings

        assert settings.max_documents_per_user == 0
        r = client.post(
            "/api/ingest",
            files={"file": ("a.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
        assert r.status_code == 200

    def test_document_count_limit_is_enforced(self, client, mock_rag, monkeypatch):
        from docusense.config.settings import settings

        monkeypatch.setattr(settings, "max_documents_per_user", 1)
        # The fixture's user already owns one document.
        r = client.post(
            "/api/ingest",
            files={"file": ("second.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
        assert r.status_code == 409
        assert "limit" in r.json()["detail"].lower()
        mock_rag.ingest.assert_not_called()

    def test_oversized_upload_is_refused(self, client, mock_rag, monkeypatch):
        from docusense.config.settings import settings

        monkeypatch.setattr(settings, "max_file_size_mb", 1)
        r = client.post(
            "/api/ingest",
            files={"file": ("big.pdf", io.BytesIO(b"x" * 2 * 1024 * 1024),
                            "application/pdf")},
        )
        assert r.status_code == 413
        mock_rag.ingest.assert_not_called()


class TestAskEndpoint:
    def test_ask_question(self, client):
        """Test asking a question."""
        response = client.post("/api/ask", json={
            "query": "What is BERT?",
            "top_k": 5,
            "mode": "answer"
        })
        assert response.status_code == 200
        data = response.json()
        assert "BERT" in data["answer"]
        assert data["confidence"] > 0

    def test_ask_compare(self, client):
        """Test compare mode."""
        response = client.post("/api/ask", json={
            "query": "Compare BERT and GPT",
            "mode": "compare"
        })
        assert response.status_code == 200

    def test_ask_empty_query(self, client):
        """Test empty query validation."""
        response = client.post("/api/ask", json={"query": ""})
        assert response.status_code == 422  # Validation error


class TestChatEndpoints:
    def test_start_chat(self, client):
        """Test starting a chat."""
        response = client.post("/api/chat/start", json={"title": "BERT Chat"})
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "conv_test123"

    def test_chat_message(self, client):
        """Test sending a chat message."""
        response = client.post("/api/chat/conv_test123", json={
            "query": "What is BERT?"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "conv_test123"
        assert data["turn_number"] == 1

    def test_chat_history(self, client):
        """Test getting chat history."""
        response = client.get("/api/chat/conv_test123")
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 2

    def test_list_chats(self, client):
        """Test listing conversations."""
        response = client.get("/api/chats")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Test Chat"


class TestDocumentEndpoints:
    def test_list_documents(self, client):
        """Test listing documents."""
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["documents"][0]["filename"] == "bert.pdf"

    def test_delete_document(self, client):
        """Test deleting a document."""
        response = client.delete("/api/documents/doc_1")
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_delete_not_found(self, client, mock_rag):
        """Test deleting non-existent document."""
        mock_rag.delete_document.return_value = False
        response = client.delete("/api/documents/nonexistent")
        assert response.status_code == 404


class TestSystemEndpoints:
    def test_health(self, client):
        """Test health check."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_stats(self, client):
        """Test query stats."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_queries"] == 5


# ==============================================================================
# What the marketing pages are allowed to claim
# ==============================================================================

class TestHealthReportsTheGenerationBackend:
    """
    The landing and sign-in pages state that generation is local and that a
    document never leaves the machine. That is true of a local install and
    false of a deployment pointed at a hosted model — and one of those
    statements sits directly above a password field. The pages correct
    themselves from this field, so it has to be right.
    """

    def test_ollama_reports_local(self, client, monkeypatch):
        from docusense.config.settings import settings

        monkeypatch.setattr(settings, "llm_provider", "ollama")
        assert client.get("/api/health").json()["generation"] == "local"

    def test_groq_reports_hosted(self, client, monkeypatch):
        from docusense.config.settings import settings

        monkeypatch.setattr(settings, "llm_provider", "groq")
        assert client.get("/api/health").json()["generation"] == "hosted"

    def test_an_unknown_provider_is_not_assumed_local(self, client, monkeypatch):
        """
        Anything that is not Ollama sends the text somewhere. Defaulting the
        unknown case to "local" would be the one wrong answer here.
        """
        from docusense.config.settings import settings

        monkeypatch.setattr(settings, "llm_provider", "something-new")
        assert client.get("/api/health").json()["generation"] == "hosted"


# ==============================================================================
# What a deployment exposes by URL
# ==============================================================================

class TestInteractiveDocsAreDevOnly:
    """
    The landing page no longer links /docs, but an unlinked route is still a
    served route. FastAPI's Swagger UI and ReDoc pages load their JavaScript
    from cdn.jsdelivr.net onto the app's own origin, which is where the
    session token lives, so they are off when ENVIRONMENT=prod.

    These build the app the way uvicorn does rather than inspecting the
    setting, because the setting being right proves nothing about the routes.
    """

    @staticmethod
    def _client(monkeypatch, environment):
        from fastapi.testclient import TestClient
        from docusense.config.settings import settings
        from docusense.api.app import create_app

        monkeypatch.setattr(settings, "environment", environment)
        # No `with`: the real lifespan builds the whole RAG stack, and none of
        # these three routes needs it.
        return TestClient(create_app())

    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
    def test_served_in_development(self, monkeypatch, path):
        assert self._client(monkeypatch, "dev").get(path).status_code == 200

    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
    def test_not_served_in_production(self, monkeypatch, path):
        assert self._client(monkeypatch, "prod").get(path).status_code == 404

    def test_the_api_itself_is_unaffected(self, monkeypatch):
        """
        Turning the docs off must not turn anything else off. The landing
        page redirect is the cheapest route that proves the app still serves.
        """
        r = self._client(monkeypatch, "prod").get("/", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/static/index.html"
