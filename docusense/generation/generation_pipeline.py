"""
Generation Pipeline - End-to-end query → answer with citations.

This is Step 4 of Phase 4: Answer Generation with Citations (FINAL).

PURPOSE:
--------
Orchestrate the complete answer generation flow:
1. Retrieval (via RetrievalPipeline)
2. Context building with academic metadata
3. Answer generation (via AnswerGenerator)
4. Citation formatting (via CitationFormatter)
5. Return complete PipelineResponse

PIPELINE ARCHITECTURE:
----------------------
                    User Query
                        ↓
            ┌──────────────────────┐
            │  Retrieval Pipeline  │  Query processing + search
            │  (Phase 3)           │
            └──────────────────────┘
                        ↓
            ┌──────────────────────┐
            │  Answer Generator    │  LLM-based answer with citations
            │  (Ollama/Llama 3.2)  │
            └──────────────────────┘
                        ↓
            ┌──────────────────────┐
            │  Citation Formatter  │  APA, reference list, BibTeX
            │  (Academic style)    │
            └──────────────────────┘
                        ↓
                PipelineResponse
                (answer + citations + metrics)

USAGE EXAMPLE:
--------------
```python
from docusense.generation import GenerationPipeline

pipeline = GenerationPipeline()
response = pipeline.generate("What F1 score did BERT achieve on SST-2?")

print(response.answer)
# "BERT achieved 93.5% F1 on SST-2 (Devlin et al., 2018, Results)"

print(response.reference_list)
# [1] Devlin, J., et al. (2018). BERT: Pre-training of Deep...

print(response.bibtex)
# @inproceedings{devlin2018bert, ...}
```

Author: DocuSense
Created: 2026-03-06
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import time

from loguru import logger

from docusense.llms.base import LLMClient
from docusense.llms.factory import get_llm_client
from docusense.generation.answer_generator import AnswerGenerator, GeneratedAnswer
from docusense.generation.citation_formatter import (
    CitationFormatter,
    CitationStyle,
    FormattedCitation
)

if TYPE_CHECKING:
    from docusense.retrieval.retrieval_pipeline import RetrievalPipeline


@dataclass
class PipelineResponse:
    """
    Complete response from the generation pipeline.
    
    Contains:
    - Generated answer with inline citations
    - Formatted reference list
    - BibTeX export
    - Performance metrics
    - Source tracking
    """
    # Core answer
    query: str
    answer: str
    
    # Citations
    citations: List[FormattedCitation] = field(default_factory=list)
    reference_list: str = ""
    bibtex: str = ""
    
    # Sources
    sources: List[Dict[str, Any]] = field(default_factory=list)
    papers_cited: List[str] = field(default_factory=list)
    num_sources: int = 0
    
    # Quality
    confidence: float = 0.0
    has_citations: bool = False
    is_multi_paper: bool = False
    
    # Performance
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    model_used: str = ""
    
    # Metadata
    retrieval_mode: str = ""
    generation_mode: str = ""
    
    def __str__(self) -> str:
        """Human-readable response."""
        parts = [
            f"Question: {self.query}",
            "",
            f"Answer: {self.answer}",
        ]
        
        if self.reference_list:
            parts.append(self.reference_list)
        
        parts.append("")
        parts.append(
            f"[{self.num_sources} sources, {len(self.papers_cited)} papers, "
            f"confidence: {self.confidence:.2f}, time: {self.total_time:.2f}s]"
        )
        
        return "\n".join(parts)


class GenerationPipeline:
    """
    End-to-end pipeline: query → answer with citations.
    
    Orchestrates:
    1. Retrieval (optional - can accept pre-retrieved results)
    2. Answer generation with academic prompts
    3. Citation formatting
    4. Response assembly
    
    Features:
    - Works with or without retrieval pipeline
    - Supports pre-retrieved results
    - Multiple citation styles
    - Comparison and conflict detection modes
    - Performance tracking
    
    Usage:
        # With retrieval (full pipeline)
        pipeline = GenerationPipeline(retrieval_pipeline=retrieval)
        response = pipeline.generate("What is BERT?")
        
        # Without retrieval (provide results directly)
        pipeline = GenerationPipeline()
        response = pipeline.generate_from_results(query, retrieval_results)
    """
    
    def __init__(
        self,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
        llm_client: Optional[LLMClient] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        citation_formatter: Optional[CitationFormatter] = None,
        citation_style: CitationStyle = CitationStyle.INLINE,
        include_references: bool = True,
        include_bibtex: bool = True
    ):
        """
        Initialize GenerationPipeline.
        
        Args:
            retrieval_pipeline: RetrievalPipeline for document search
            llm_client: Generation backend (built from LLM_PROVIDER if None)
            answer_generator: AnswerGenerator instance
            citation_formatter: CitationFormatter instance
            citation_style: Default citation style
            include_references: Include reference list in response
            include_bibtex: Include BibTeX in response
        """
        self.retrieval_pipeline = retrieval_pipeline
        self.citation_style = citation_style
        self.include_references = include_references
        self.include_bibtex = include_bibtex
        
        # Initialize components
        self.client = llm_client or get_llm_client()
        self.answer_generator = answer_generator or AnswerGenerator(
            llm_client=self.client
        )
        self.citation_formatter = citation_formatter or CitationFormatter()
        
        logger.success("🚀 GenerationPipeline initialized")
        logger.info(f"  Retrieval: {'connected' if retrieval_pipeline else 'manual mode'}")
        logger.info(f"  Citation style: {citation_style.value}")
        logger.info(f"  References: {include_references}")
        logger.info(f"  BibTeX: {include_bibtex}")
    
    def generate(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        mode: str = "answer"
    ) -> PipelineResponse:
        """
        Full pipeline: query → retrieval → answer with citations.
        
        Requires a retrieval_pipeline to be configured.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters for retrieval
            context: Optional conversation context
            mode: "answer" (default), "compare" (multi-paper), "conflicts"
            
        Returns:
            PipelineResponse with answer, citations, and metrics
        """
        if not self.retrieval_pipeline:
            raise RuntimeError(
                "No retrieval pipeline configured. "
                "Use generate_from_results() with pre-retrieved results, "
                "or provide a RetrievalPipeline at initialization."
            )
        
        start_time = time.time()
        
        logger.info(f"🚀 Full pipeline: '{query}'")
        logger.info(f"  Mode: {mode}, Top-K: {top_k}")
        
        # Step 1: Retrieval
        retrieval_start = time.time()
        try:
            results, metrics = self.retrieval_pipeline.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
                context=context
            )
            retrieval_time = time.time() - retrieval_start
            logger.info(f"  📚 Retrieved {len(results)} chunks in {retrieval_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return PipelineResponse(
                query=query,
                answer=f"Retrieval failed: {e}. Unable to find relevant documents.",
                total_time=time.time() - start_time,
                retrieval_mode="failed"
            )
        
        if not results:
            return PipelineResponse(
                query=query,
                answer="No relevant documents found for your query. "
                       "Please try rephrasing or broadening your search.",
                retrieval_time=retrieval_time,
                total_time=time.time() - start_time,
                retrieval_mode=getattr(metrics, 'stages_used', ['none'])[0] if hasattr(metrics, 'stages_used') and metrics.stages_used else 'none'
            )
        
        # Step 2: Generate answer from results
        response = self.generate_from_results(
            query=query,
            retrieval_results=results,
            context=context,
            mode=mode
        )
        
        # Update timing
        response.retrieval_time = retrieval_time
        response.total_time = time.time() - start_time
        response.retrieval_mode = ", ".join(getattr(metrics, 'stages_used', []))
        
        logger.success(
            f"✅ Pipeline complete: {response.total_time:.2f}s total "
            f"(retrieval: {response.retrieval_time:.2f}s, "
            f"generation: {response.generation_time:.2f}s)"
        )
        
        return response
    
    def generate_stream(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
    ):
        """
        Run the pipeline, streaming the answer as the model writes it.

        Retrieval completes first (it is fast and its result is needed to build
        the prompt), then answer text is yielded incrementally. Citations are
        formatted at the end, once the full answer text exists.

        Yields:
            ("status", str)      progress before any text is available
            ("token", str)       answer fragments, in order
            ("done", PipelineResponse)  final answer plus citations and metrics
            ("error", str)       terminal failure; no "done" follows
        """
        start_time = time.time()
        logger.info(f"🌊 Streaming pipeline: '{query}' (top_k={top_k})")

        # Step 1: Retrieval
        retrieval_start = time.time()
        yield ("status", "Searching your documents...")
        try:
            results, metrics = self.retrieval_pipeline.retrieve(
                query=query, top_k=top_k, filters=filters, context=context
            )
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            yield ("error", f"Retrieval failed: {e}")
            return

        retrieval_time = time.time() - retrieval_start

        if not results:
            yield ("done", PipelineResponse(
                query=query,
                answer="No relevant documents found for your query. "
                       "Please try rephrasing or broadening your search.",
                retrieval_time=retrieval_time,
                total_time=time.time() - start_time,
            ))
            return

        yield ("status", f"Reading {len(results)} sources...")

        # Step 2: Stream the answer
        generated = None
        for kind, payload in self.answer_generator.generate_answer_stream(
            query=query, retrieval_results=results, context=context
        ):
            if kind == "token":
                yield ("token", payload)
            elif kind == "error":
                yield ("error", payload)
            elif kind == "done":
                generated = payload

        if generated is None:
            yield ("error", "Generation produced no result")
            return

        # Step 3: Citations, once the whole answer exists
        response = self._build_response(generated)
        response.retrieval_time = retrieval_time
        response.total_time = time.time() - start_time
        response.retrieval_mode = ", ".join(getattr(metrics, "stages_used", []))

        yield ("done", response)

    def generate_from_results(
        self,
        query: str,
        retrieval_results: List[Any],
        context: Optional[str] = None,
        mode: str = "answer"
    ) -> PipelineResponse:
        """
        Generate answer from pre-retrieved results (no retrieval step).
        
        Args:
            query: User's question
            retrieval_results: Pre-retrieved RetrievalResult objects
            context: Optional conversation context
            mode: "answer", "compare", or "conflicts"
            
        Returns:
            PipelineResponse with answer, citations, and metrics
        """
        start_time = time.time()
        
        logger.info(f"📝 Generating from {len(retrieval_results)} results (mode: {mode})")
        
        # Step 1: Generate answer based on mode
        gen_start = time.time()
        
        if mode == "compare":
            generated = self.answer_generator.compare_papers(query, retrieval_results)
        elif mode == "conflicts":
            generated = self.answer_generator.detect_conflicts(retrieval_results)
        else:
            generated = self.answer_generator.generate_answer(
                query, retrieval_results, context
            )
        
        generation_time = time.time() - gen_start

        response = self._build_response(generated, mode=mode)
        response.generation_time = generation_time
        response.total_time = time.time() - start_time

        logger.success(
            f"✅ Generated answer: {len(generated.answer)} chars, "
            f"{len(response.citations)} citations, {generation_time:.2f}s"
        )

        return response

    def _build_response(
        self,
        generated: GeneratedAnswer,
        mode: str = "answer",
    ) -> PipelineResponse:
        """
        Format citations and assemble a PipelineResponse from a generated answer.

        Shared by the buffered and streaming paths so both produce identical
        citations, reference lists, and BibTeX for the same answer.
        """
        citations = []
        reference_list = ""
        bibtex = ""

        if generated.sources:
            citations = self.citation_formatter.format_sources(
                generated.sources,
                style=self.citation_style
            )

            if self.include_references:
                reference_list = self.citation_formatter.format_reference_list(
                    generated.sources
                )

            if self.include_bibtex:
                bibtex = self.citation_formatter.format_bibtex_export(
                    generated.sources
                )

        return PipelineResponse(
            query=generated.query,
            answer=generated.answer,
            citations=citations,
            reference_list=reference_list,
            bibtex=bibtex,
            sources=generated.sources,
            papers_cited=generated.papers_cited,
            num_sources=generated.num_sources,
            confidence=generated.confidence,
            has_citations=generated.has_citations,
            is_multi_paper=generated.is_multi_paper,
            generation_time=generated.generation_time,
            model_used=generated.model_used,
            generation_mode=mode
        )
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return {
            "retrieval_connected": self.retrieval_pipeline is not None,
            "citation_style": self.citation_style.value,
            "include_references": self.include_references,
            "include_bibtex": self.include_bibtex,
            "model": self.client.model,
            "ollama_available": self.client.is_available()
        }
