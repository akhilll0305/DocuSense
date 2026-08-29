"""
Tests for Conversation Store and Conversation Manager.

Phase 5: Complete RAG Pipeline — conversation memory tests.
Uses SQLite :memory: for fast, isolated tests.
"""

from pathlib import Path
from unittest.mock import MagicMock


# ==============================================================================
# ConversationStore Tests
# ==============================================================================

class TestConversationStore:
    """Tests for ConversationStore using in-memory SQLite."""

    def _make_store(self):
        """Create a store with an in-memory DB."""
        from docusense.storage.conversation_store import ConversationStore
        # Use a temp file path (ConversationStore will create it)
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "test_conv.db"
        store = ConversationStore(db_path=tmp)
        return store

    def test_create_conversation(self):
        """Test creating a conversation."""
        store = self._make_store()
        try:
            conv_id = store.create_conversation("BERT Analysis")
            assert conv_id.startswith("conv_")
            assert len(conv_id) > 5

            conv = store.get_conversation(conv_id)
            assert conv is not None
            assert conv.title == "BERT Analysis"
            assert conv.conversation_id == conv_id
        finally:
            store.close()

    def test_list_conversations(self):
        """Test listing conversations."""
        store = self._make_store()
        try:
            store.create_conversation("First")
            store.create_conversation("Second")
            store.create_conversation("Third")

            convs = store.list_conversations()
            assert len(convs) == 3
            # Most recent first
            assert convs[0].title == "Third"
        finally:
            store.close()

    def test_delete_conversation(self):
        """Test deleting a conversation cascades to messages."""
        store = self._make_store()
        try:
            conv_id = store.create_conversation("To Delete")
            store.add_message(conv_id, "user", "Hello")
            store.add_message(conv_id, "assistant", "Hi there")

            assert store.delete_conversation(conv_id) is True
            assert store.get_conversation(conv_id) is None
            assert store.get_messages(conv_id) == []
        finally:
            store.close()

    def test_add_and_get_messages(self):
        """Test adding and retrieving messages."""
        store = self._make_store()
        try:
            conv_id = store.create_conversation("Test")
            
            msg1 = store.add_message(conv_id, "user", "What is BERT?")
            msg2 = store.add_message(
                conv_id, "assistant",
                "BERT is a bidirectional transformer.",
                sources=[{"paper_title": "BERT Paper"}]
            )

            assert msg1.startswith("msg_")
            assert msg2.startswith("msg_")

            messages = store.get_messages(conv_id)
            assert len(messages) == 2
            assert messages[0].role == "user"
            assert messages[0].content == "What is BERT?"
            assert messages[1].role == "assistant"
            assert messages[1].sources[0]["paper_title"] == "BERT Paper"
        finally:
            store.close()

    def test_get_messages_with_limit(self):
        """Test message retrieval with limit (most recent)."""
        store = self._make_store()
        try:
            conv_id = store.create_conversation("Test")
            for i in range(10):
                store.add_message(conv_id, "user", f"Message {i}")

            recent = store.get_messages(conv_id, limit=3)
            assert len(recent) == 3
            # Should be the 3 most recent, in chronological order
            assert "Message 7" in recent[0].content
        finally:
            store.close()

    def test_log_query(self):
        """Test query logging."""
        store = self._make_store()
        try:
            qid = store.log_query(
                query="What is BERT?",
                processed_query="what bert",
                num_results=5,
                response_time=1.23,
                confidence=0.85,
                model_used="llama3.2:3b"
            )

            assert qid.startswith("qry_")

            history = store.get_query_history(limit=10)
            assert len(history) == 1
            assert history[0].query == "What is BERT?"
            assert history[0].response_time == 1.23
            assert history[0].confidence == 0.85
        finally:
            store.close()

    def test_query_stats(self):
        """Test aggregate query statistics."""
        store = self._make_store()
        try:
            store.log_query("Q1", response_time=1.0, confidence=0.8)
            store.log_query("Q2", response_time=2.0, confidence=0.9)
            store.log_query("Q3", response_time=3.0, confidence=0.7)

            stats = store.get_query_stats()
            assert stats["total_queries"] == 3
            assert stats["avg_response_time"] == 2.0
        finally:
            store.close()

    def test_empty_query_stats(self):
        """Test stats with no queries."""
        store = self._make_store()
        try:
            stats = store.get_query_stats()
            assert stats["total_queries"] == 0
        finally:
            store.close()

    def test_conversation_not_found(self):
        """Test getting a non-existent conversation."""
        store = self._make_store()
        try:
            assert store.get_conversation("conv_nonexistent") is None
        finally:
            store.close()


# ==============================================================================
# ConversationManager Tests
# ==============================================================================

class TestConversationManager:
    """Tests for ConversationManager."""

    def _make_manager(self, with_pipeline=False):
        """Create a manager with in-memory store."""
        from docusense.generation.conversation_manager import ConversationManager
        from docusense.storage.conversation_store import ConversationStore
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "test_mgr.db"
        store = ConversationStore(db_path=tmp)

        pipeline = None
        if with_pipeline:
            pipeline = MagicMock()
            pipeline.retrieval_pipeline = MagicMock()
            mock_response = MagicMock()
            mock_response.answer = "BERT is a transformer model (Devlin et al., 2018)."
            mock_response.sources = [{"paper_title": "BERT Paper"}]
            mock_response.papers_cited = ["BERT Paper"]
            mock_response.reference_list = "[1] Devlin, J. (2018). BERT."
            mock_response.confidence = 0.85
            pipeline.generate.return_value = mock_response
            pipeline.client = MagicMock()
            pipeline.client.model = "llama3.2:3b"

        manager = ConversationManager(
            generation_pipeline=pipeline,
            conversation_store=store
        )
        return manager

    def test_start_conversation(self):
        """Test starting a conversation."""
        manager = self._make_manager()
        try:
            conv_id = manager.start_conversation("Test Chat")
            assert conv_id.startswith("conv_")

            convs = manager.list_conversations()
            assert len(convs) == 1
            assert convs[0].title == "Test Chat"
        finally:
            manager.close()

    def test_chat_without_pipeline(self):
        """Test chat when no pipeline is configured."""
        manager = self._make_manager(with_pipeline=False)
        try:
            conv_id = manager.start_conversation("Test")
            response = manager.chat(conv_id, "What is BERT?")

            assert "not configured" in response.answer
            assert response.conversation_id == conv_id
            assert response.turn_number == 1
        finally:
            manager.close()

    def test_chat_with_pipeline(self):
        """Test chat with a mocked pipeline."""
        manager = self._make_manager(with_pipeline=True)
        try:
            conv_id = manager.start_conversation("BERT Chat")
            response = manager.chat(conv_id, "What is BERT?")

            assert "BERT" in response.answer
            assert response.confidence == 0.85
            assert response.papers_cited == ["BERT Paper"]
            assert response.turn_number == 1

            # Check messages were saved
            history = manager.get_history(conv_id)
            assert len(history) == 2  # user + assistant
            assert history[0].role == "user"
            assert history[1].role == "assistant"
        finally:
            manager.close()

    def test_multi_turn_chat(self):
        """Test multi-turn conversation builds context."""
        manager = self._make_manager(with_pipeline=True)
        try:
            conv_id = manager.start_conversation("Multi-turn")
            
            r1 = manager.chat(conv_id, "What is BERT?")
            r2 = manager.chat(conv_id, "What about its accuracy?")

            assert r1.turn_number == 1
            assert r2.turn_number == 2

            history = manager.get_history(conv_id)
            assert len(history) == 4  # 2 user + 2 assistant

            # Verify context was passed to pipeline
            calls = manager.pipeline.generate.call_args_list
            assert calls[1].kwargs.get("context") is not None or (
                len(calls[1].args) > 3 and calls[1].args[3] is not None
            )
        finally:
            manager.close()

    def test_build_context(self):
        """Test context building from history."""
        from docusense.generation.conversation_manager import ConversationManager
        from docusense.storage.conversation_store import Message

        manager = ConversationManager(context_window=4)
        
        history = [
            Message("m1", "c1", "user", "What is BERT?"),
            Message("m2", "c1", "assistant", "BERT is a transformer model."),
        ]

        context = manager._build_context(history)
        assert "User: What is BERT?" in context
        assert "Assistant: BERT is a transformer model." in context
        manager.close()

    def test_build_context_truncation(self):
        """Test context truncation for long messages."""
        from docusense.generation.conversation_manager import ConversationManager
        from docusense.storage.conversation_store import Message

        manager = ConversationManager(max_context_chars=100)

        history = [
            Message("m1", "c1", "user", "A" * 200),
            Message("m2", "c1", "assistant", "B" * 200),
        ]

        context = manager._build_context(history)
        # Should be truncated
        assert len(context) < 1000
        manager.close()

    def test_delete_conversation(self):
        """Test deleting a conversation."""
        manager = self._make_manager()
        try:
            conv_id = manager.start_conversation("To Delete")
            assert manager.delete_conversation(conv_id) is True
            assert manager.list_conversations() == []
        finally:
            manager.close()

    def test_query_stats(self):
        """Test query analytics through manager."""
        manager = self._make_manager(with_pipeline=True)
        try:
            conv_id = manager.start_conversation("Stats Test")
            manager.chat(conv_id, "What is BERT?")
            manager.chat(conv_id, "How does it work?")

            stats = manager.get_query_stats()
            assert stats["total_queries"] == 2
        finally:
            manager.close()
