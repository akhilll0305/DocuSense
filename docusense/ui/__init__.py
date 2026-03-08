"""
DocuSense UI Module.

Phase 7: Gradio interface for the RAG system.

Run: python -m docusense.ui.gradio_app
"""

from .gradio_app import DocuSenseUI, create_app

__all__ = ["DocuSenseUI", "create_app"]
