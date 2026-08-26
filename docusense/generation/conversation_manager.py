"""
Conversation Manager - Multi-turn conversation context for RAG.

Phase 5: Complete RAG Pipeline (Step 2)

PURPOSE:
--------
Manage multi-turn conversations with context:
1. Track conversation history
2. Build LLM context from recent messages
3. Support follow-up questions ("What about their methodology?")
4. Log queries for analytics
5. Persist everything to SQLite

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from loguru import logger

from docusense.storage.conversation_store import (
    ConversationStore,
    Conversation,
    Message,
    QueryLog
)

if TYPE_CHECKING:
    from docusense.generation.generation_pipeline import GenerationPipeline, PipelineResponse


@dataclass
class ChatResponse:
    """Response from a chat turn."""
    answer: str
    conversation_id: str
    message_id: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    papers_cited: List[str] = field(default_factory=list)
    reference_list: str = ""
    confidence: float = 0.0
    response_time: float = 0.0
    turn_number: int = 0


class ConversationManager:
    """
    Manage multi-turn conversations with context.

    Features:
    - Sliding window of recent messages for context
    - Automatic follow-up question resolution
    - Query logging and analytics
    - Persistent conversation storage

    Usage:
        manager = ConversationManager(generation_pipeline=pipeline)
        
        # Start a conversation
        conv_id = manager.start_conversation("BERT Analysis")
        
        # Chat with context
        response = manager.chat(conv_id, "What is BERT?")
        response = manager.chat(conv_id, "What about its accuracy?")  # follows up
    """

    def __init__(
        self,
        generation_pipeline: Optional[GenerationPipeline] = None,
        conversation_store: Optional[ConversationStore] = None,
        context_window: int = 6,
        max_context_chars: int = 4000
    ):
        """
        Initialize ConversationManager.

        Args:
            generation_pipeline: Pipeline for generating answers
            conversation_store: Store for persistence (creates new if None)
            context_window: Number of recent messages to include as context
            max_context_chars: Max characters for context string
        """
        self.pipeline = generation_pipeline
        self.store = conversation_store or ConversationStore()
        self.context_window = context_window
        self.max_context_chars = max_context_chars

        logger.info(f"💬 ConversationManager initialized (window: {context_window})")

    def start_conversation(
        self,
        title: str = "New Conversation",
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Start a new conversation.

        Args:
            title: Display title
            metadata: Arbitrary JSON metadata
            user_id: Owning user

        Returns:
            conversation_id
        """
        conv_id = self.store.create_conversation(title, metadata, user_id=user_id)
        logger.info(f"💬 Started conversation: {conv_id} ({title})")
        return conv_id

    def chat(
        self,
        conversation_id: str,
        query: str,
        mode: str = "answer",
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """
        Process a chat turn with conversation context.

        Steps:
        1. Get recent message history
        2. Build context string for LLM
        3. Generate answer using pipeline
        4. Save user message and assistant response
        5. Log query for analytics

        Args:
            conversation_id: Active conversation ID
            query: User's question
            mode: "answer", "compare", or "conflicts"
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters

        Returns:
            ChatResponse with answer and metadata
        """
        import time
        start_time = time.time()

        logger.info(f"💬 Chat turn in {conversation_id}: '{query}'")

        # Step 1: Get conversation history
        history = self.store.get_messages(conversation_id, limit=self.context_window)
        turn_number = len(history) // 2 + 1  # Approximate turn count

        # Step 2: Build context from history
        context = self._build_context(history) if history else None

        # Step 3: Save user message
        user_msg_id = self.store.add_message(conversation_id, "user", query)

        # Step 4: Generate answer
        if self.pipeline:
            try:
                if self.pipeline.retrieval_pipeline:
                    response = self.pipeline.generate(
                        query=query,
                        top_k=top_k,
                        filters=filters,
                        context=context,
                        mode=mode
                    )
                else:
                    # No retrieval — generate from empty results
                    response = self.pipeline.generate_from_results(
                        query=query,
                        retrieval_results=[],
                        context=context,
                        mode=mode
                    )

                answer = response.answer
                sources = response.sources
                papers = response.papers_cited
                ref_list = response.reference_list
                confidence = response.confidence
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")
                answer = f"I encountered an error generating a response: {e}"
                sources = []
                papers = []
                ref_list = ""
                confidence = 0.0
        else:
            answer = (
                "Generation pipeline not configured. "
                "Please provide a GenerationPipeline at initialization."
            )
            sources = []
            papers = []
            ref_list = ""
            confidence = 0.0

        elapsed = time.time() - start_time

        # Step 5: Save assistant message
        asst_msg_id = self.store.add_message(
            conversation_id, "assistant", answer,
            sources=sources,
            metadata={"papers_cited": papers, "confidence": confidence}
        )

        # Step 6: Log query
        self.store.log_query(
            query=query,
            num_results=len(sources),
            response_time=elapsed,
            confidence=confidence,
            model_used=getattr(self.pipeline, 'client', None) and self.pipeline.client.model or "",
            conversation_id=conversation_id
        )

        logger.success(f"💬 Turn {turn_number} complete ({elapsed:.2f}s)")

        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            message_id=asst_msg_id,
            sources=sources,
            papers_cited=papers,
            reference_list=ref_list,
            confidence=confidence,
            response_time=elapsed,
            turn_number=turn_number
        )

    def chat_stream(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ):
        """
        Process a chat turn, yielding the answer as it is generated.

        Persists the user message before generating and the assistant message
        after, so a streamed turn leaves the same history as a buffered one.

        Yields:
            (kind, payload) — "status", "token", "error", then a final "done"
            carrying a ChatResponse.
        """
        import time
        start_time = time.time()

        logger.info(f"💬 Streaming chat turn in {conversation_id}: '{query}'")

        history = self.store.get_messages(conversation_id, limit=self.context_window)
        turn_number = len(history) // 2 + 1
        context = self._build_context(history) if history else None

        self.store.add_message(conversation_id, "user", query)

        if not self.pipeline or not self.pipeline.retrieval_pipeline:
            yield ("error", "Generation pipeline not configured.")
            return

        response = None
        for kind, payload in self.pipeline.generate_stream(
            query=query, top_k=top_k, filters=filters, context=context
        ):
            if kind == "done":
                response = payload
            else:
                yield (kind, payload)

        if response is None:
            # generate_stream already emitted an error; record nothing further.
            return

        elapsed = time.time() - start_time

        asst_msg_id = self.store.add_message(
            conversation_id, "assistant", response.answer,
            sources=response.sources,
            metadata={
                "papers_cited": response.papers_cited,
                "confidence": response.confidence,
            }
        )

        self.store.log_query(
            query=query,
            num_results=len(response.sources),
            response_time=elapsed,
            confidence=response.confidence,
            model_used=getattr(self.pipeline, 'client', None) and self.pipeline.client.model or "",
            conversation_id=conversation_id
        )

        logger.success(f"💬 Streamed turn {turn_number} complete ({elapsed:.2f}s)")

        yield ("done", ChatResponse(
            answer=response.answer,
            conversation_id=conversation_id,
            message_id=asst_msg_id,
            sources=response.sources,
            papers_cited=response.papers_cited,
            reference_list=response.reference_list,
            confidence=response.confidence,
            response_time=elapsed,
            turn_number=turn_number
        ))

    def get_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get conversation message history."""
        return self.store.get_messages(conversation_id, limit)

    def list_conversations(
        self, limit: int = 20, user_id: Optional[str] = None
    ) -> List[Conversation]:
        """List recent conversations, optionally scoped to one owner."""
        return self.store.list_conversations(limit, user_id=user_id)

    def get_owner(self, conversation_id: str) -> Optional[str]:
        """Return the user_id owning a conversation."""
        return self.store.get_conversation_owner(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        return self.store.delete_conversation(conversation_id)

    def get_query_stats(self) -> Dict[str, Any]:
        """Get query analytics."""
        return self.store.get_query_stats()

    # ==================================================================
    # CONTEXT BUILDING
    # ==================================================================

    def _build_context(self, history: List[Message]) -> str:
        """
        Build conversation context string from message history.

        Creates a summary of recent exchanges for the LLM to understand
        the conversation flow and handle follow-up questions.
        """
        if not history:
            return ""

        parts = ["Previous conversation:"]

        total_chars = 0
        for msg in history:
            role_label = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages in context
            content = msg.content
            if len(content) > 500:
                content = content[:500] + "..."

            line = f"{role_label}: {content}"
            if total_chars + len(line) > self.max_context_chars:
                break
            parts.append(line)
            total_chars += len(line)

        return "\n".join(parts)

    def close(self):
        """Close the conversation store."""
        self.store.close()
