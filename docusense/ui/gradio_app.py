"""
Gradio UI - Interactive web interface for DocuSense.

Phase 7: API & UI (Step 4)

Features:
---------
1. Document Upload: Drag-and-drop PDF/DOCX ingestion
2. Ask Questions: Single-shot Q&A with citations
3. Chat: Multi-turn conversation with memory
4. Document Library: View and manage ingested papers
5. Benchmarks: Run evaluation benchmarks

Run with: python -m docusense.ui.gradio_app

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from pathlib import Path

from loguru import logger

# Lazy gradio import
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None


class DocuSenseUI:
    """
    Gradio-based web UI for DocuSense.

    Usage:
        ui = DocuSenseUI()
        ui.launch()  # Opens browser at http://localhost:7860
    """

    def __init__(self, rag=None):
        """
        Initialize the UI.

        Args:
            rag: Optional DocuSenseRAG instance (creates one if None)
        """
        if not GRADIO_AVAILABLE:
            raise ImportError(
                "Gradio is required for the UI. Install with: pip install gradio>=4.11.0"
            )

        self.rag = rag
        self._active_conversation = None

        logger.info("🖥️ DocuSenseUI initialized")

    def _get_rag(self):
        """Lazily initialize RAG."""
        if self.rag is None:
            from docusense.rag_pipeline import DocuSenseRAG
            self.rag = DocuSenseRAG()
        return self.rag

    # ==================================================================
    # HANDLERS
    # ==================================================================

    def handle_upload(self, file) -> str:
        """Handle document upload."""
        if file is None:
            return "⚠️ No file selected."

        try:
            rag = self._get_rag()
            file_path = file.name if hasattr(file, 'name') else str(file)
            result = rag.ingest(file_path)

            if result.success:
                paper_info = f"\n📄 **Paper:** {result.paper_title}" if result.paper_title else ""
                return (
                    f"✅ **Ingested:** {result.filename}\n"
                    f"📊 **Chunks:** {result.num_chunks} | "
                    f"**Embeddings:** {result.num_embeddings}\n"
                    f"⏱️ **Time:** {result.processing_time:.1f}s"
                    f"{paper_info}"
                )
            else:
                return f"❌ **Failed:** {result.error}"
        except Exception as e:
            return f"❌ **Error:** {str(e)}"

    def handle_ask(self, query: str, mode: str, top_k: int) -> str:
        """Handle a question."""
        if not query.strip():
            return "⚠️ Please enter a question."

        try:
            rag = self._get_rag()
            response = rag.ask(query=query, mode=mode.lower(), top_k=int(top_k))

            parts = [f"### Answer\n{response.answer}"]

            if response.reference_list:
                parts.append(f"\n### References\n{response.reference_list}")

            parts.append(
                f"\n---\n"
                f"📊 Confidence: {response.confidence:.0%} | "
                f"📚 Sources: {response.num_sources} | "
                f"⏱️ Time: {response.total_time:.2f}s"
            )

            return "\n".join(parts)
        except Exception as e:
            return f"❌ **Error:** {str(e)}"

    def handle_chat(
        self,
        message: str,
        history: List[dict],
        mode: str
    ) -> Tuple[str, List[dict]]:
        """Handle a chat message (Gradio chatbot format)."""
        if not message.strip():
            return "", history

        try:
            rag = self._get_rag()

            if self._active_conversation is None:
                self._active_conversation = rag.start_chat("Gradio Chat")

            response = rag.chat(
                self._active_conversation,
                message,
                mode=mode.lower()
            )

            answer = response.answer
            if response.reference_list:
                answer += f"\n\n**References:**\n{response.reference_list}"

            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})

            return "", history
        except Exception as e:
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
            return "", history

    def handle_new_chat(self) -> Tuple[str, List[dict]]:
        """Start a new chat conversation."""
        self._active_conversation = None
        return "", []

    def handle_list_documents(self) -> str:
        """List all ingested documents."""
        try:
            rag = self._get_rag()
            docs = rag.list_documents()

            if not docs:
                return "📂 No documents ingested yet."

            lines = ["| Filename | Chunks | Paper | Title |", "|----------|--------|-------|-------|"]
            for d in docs:
                paper = "✅" if d.get("is_research_paper") else "—"
                title = d.get("paper_title", "—") or "—"
                lines.append(
                    f"| {d['filename']} | {d.get('total_chunks', 0)} | {paper} | {title} |"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def handle_run_benchmark(self) -> str:
        """Run sample benchmark."""
        try:
            from docusense.evaluation.benchmark_runner import BenchmarkRunner
            runner = BenchmarkRunner()
            report = runner.run_sample_benchmark()

            lines = [
                f"### Benchmark: {report.config.name}",
                f"**Samples:** {report.num_samples} | **Time:** {report.benchmark_time:.2f}s",
                "",
            ]

            if report.result and report.result.answer:
                a = report.result.answer
                lines.extend([
                    "| Metric | Score |",
                    "|--------|-------|",
                    f"| Token Overlap | {a.token_overlap:.4f} |",
                    f"| Citation F1 | {a.citation_f1:.4f} |",
                    f"| Completeness | {a.completeness:.4f} |",
                ])

            if report.summary:
                grade = report.summary.get("answer_grade", "N/A")
                lines.append(f"\n**Grade:** {grade}")

            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    # ==================================================================
    # BUILD UI
    # ==================================================================

    def build(self) -> gr.Blocks:
        """Build the Gradio interface."""
        with gr.Blocks(
            title="DocuSense — Research Paper Analysis",
        ) as demo:

            gr.Markdown(
                "# 📚 DocuSense\n"
                "### Research Paper Analysis RAG System\n"
                "Upload papers, ask questions, get answers with citations."
            )

            with gr.Tabs():

                # ========== Tab 1: Upload ==========
                with gr.Tab("📄 Upload", id="upload"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            upload_file = gr.File(
                                label="Upload Document",
                                file_types=[".pdf", ".docx", ".txt", ".md"],
                                type="filepath"
                            )
                            upload_btn = gr.Button("📥 Ingest", variant="primary")
                        with gr.Column(scale=2):
                            upload_output = gr.Markdown(
                                label="Result",
                                value="Upload a document to get started."
                            )

                    upload_btn.click(
                        fn=self.handle_upload,
                        inputs=[upload_file],
                        outputs=[upload_output]
                    )

                # ========== Tab 2: Ask ==========
                with gr.Tab("❓ Ask", id="ask"):
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="What F1 score did BERT achieve on SST-2?",
                            lines=2
                        )
                    with gr.Row():
                        mode_select = gr.Radio(
                            choices=["Answer", "Compare", "Conflicts"],
                            value="Answer",
                            label="Mode"
                        )
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Sources (top-K)"
                        )
                    ask_btn = gr.Button("🔍 Ask", variant="primary")
                    ask_output = gr.Markdown(label="Answer")

                    ask_btn.click(
                        fn=self.handle_ask,
                        inputs=[query_input, mode_select, top_k_slider],
                        outputs=[ask_output]
                    )

                # ========== Tab 3: Chat ==========
                with gr.Tab("💬 Chat", id="chat"):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400,
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="Message",
                            placeholder="What is BERT?",
                            scale=4
                        )
                        chat_mode = gr.Radio(
                            choices=["Answer", "Compare"],
                            value="Answer",
                            label="Mode",
                            scale=1
                        )
                    with gr.Row():
                        chat_btn = gr.Button("💬 Send", variant="primary")
                        new_chat_btn = gr.Button("🔄 New Chat")

                    chat_btn.click(
                        fn=self.handle_chat,
                        inputs=[chat_input, chatbot, chat_mode],
                        outputs=[chat_input, chatbot]
                    )

                    chat_input.submit(
                        fn=self.handle_chat,
                        inputs=[chat_input, chatbot, chat_mode],
                        outputs=[chat_input, chatbot]
                    )

                    new_chat_btn.click(
                        fn=self.handle_new_chat,
                        outputs=[chat_input, chatbot]
                    )

                # ========== Tab 4: Library ==========
                with gr.Tab("📚 Library", id="library"):
                    refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                    docs_output = gr.Markdown(value="Click Refresh to view documents.")

                    refresh_btn.click(
                        fn=self.handle_list_documents,
                        outputs=[docs_output]
                    )

                # ========== Tab 5: Benchmark ==========
                with gr.Tab("📊 Benchmark", id="benchmark"):
                    gr.Markdown(
                        "Run evaluation benchmarks to measure system quality.\n"
                        "Uses built-in sample dataset (no external data needed)."
                    )
                    bench_btn = gr.Button("🏃 Run Benchmark", variant="primary")
                    bench_output = gr.Markdown(value="Click Run Benchmark to start.")

                    bench_btn.click(
                        fn=self.handle_run_benchmark,
                        outputs=[bench_output]
                    )

        return demo

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        demo = self.build()
        demo.launch(**kwargs)


def create_app(rag=None):
    """Create and return the Gradio app."""
    ui = DocuSenseUI(rag=rag)
    return ui.build()


if __name__ == "__main__":
    ui = DocuSenseUI()
    ui.launch(server_name="0.0.0.0", server_port=7860)
