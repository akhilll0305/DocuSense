"""
Answer Generator - Citation-aware answer generation for research papers.

This is Step 2 of Phase 4: Answer Generation with Citations.

PURPOSE:
--------
Generate answers with academic citations from retrieved chunks:
1. Build context from RetrievalResult with paper metadata
2. Use academic-aware system prompts
3. Generate answers with inline citations
4. Support multi-paper comparison and conflict detection

WHY THIS MATTERS:
-----------------
- Generic RAG: "The accuracy was around 93%"
- Our RAG: "BERT achieved 93.5% ± 0.2% F1 on SST-2 (Devlin et al., 2018, Results, p.9)"

PROMPT ENGINEERING:
-------------------
The system prompt instructs the LLM to:
1. Always cite sources with (Author et al., Year, Section)
2. Include specific numbers and metrics when available
3. Distinguish between papers when multiple sources are retrieved
4. Note conflicting results across papers
5. Acknowledge limitations when evidence is insufficient

DUCK TYPING:
-------------
This module accepts any object with .chunk_id, .document_id, .text,
.score, and .metadata attributes (e.g., RetrievalResult from the
retrieval pipeline). This avoids importing the full retrieval stack
(which requires qdrant_client, sentence-transformers, etc.)

Author: DocuSense
Created: 2026-03-06
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import math
import re
import time

from loguru import logger

from docusense.config.settings import settings
from docusense.llms.ollama_client import OllamaClient

if TYPE_CHECKING:
    from docusense.retrieval.retrieval_pipeline import RetrievalResult


@dataclass
class GeneratedAnswer:
    """
    Complete answer with citations and metadata.
    
    This is what makes your RAG system ACADEMIC-GRADE!
    """
    # Core response
    query: str
    answer: str
    
    # Sources used
    sources: List[Dict[str, Any]] = field(default_factory=list)
    num_sources: int = 0
    
    # Paper info
    papers_cited: List[str] = field(default_factory=list)  # List of paper titles
    
    # Quality indicators
    confidence: float = 0.0  # 0-1: how confident in the answer
    has_citations: bool = False
    is_multi_paper: bool = False
    
    # Performance
    generation_time: float = 0.0
    model_used: str = ""
    
    def __str__(self) -> str:
        """Human-readable answer with source count."""
        return (
            f"Answer ({self.num_sources} sources, "
            f"{len(self.papers_cited)} papers):\n{self.answer}"
        )


# ==============================================================================
# SYSTEM PROMPTS
# ==============================================================================

ACADEMIC_SYSTEM_PROMPT = """You are an academic research assistant that answers questions based ONLY on the provided source documents. You must follow these rules strictly:

CITATION RULES:
1. A source block may carry a "Cite this source as:" line. When it does, copy
   that string EXACTLY, character for character, whenever you use that source.
   When a source has no such line, use its content but do NOT cite it, and
   never invent a citation or repeat the words "Author", "Year", or "Section".
2. Place the citation inline at the end of the sentence it supports, before the
   full stop. Example:
     The system combined DQN with a fuzzy inference layer (Saadi et al., 2025, methodology).
3. EVERY factual sentence drawn from the sources must carry a citation.
4. When quoting specific numbers or metrics, ALWAYS cite the source.
5. If multiple sources support the same finding, cite all of them.
6. If sources DISAGREE, explicitly note the conflict and cite each side.
7. Never invent a citation for a source that was not provided.

ANSWER RULES:
1. Answer ONLY based on the provided context - never make up information
2. If the context doesn't contain enough information, say so explicitly
3. Be precise - use exact numbers, metrics, and terminology from the papers
4. Keep answers concise but comprehensive
5. Use academic language appropriate for research discussion

FORMAT:
- Start with a direct answer to the question
- Support with evidence from the sources
- End with a brief synthesis if multiple papers are involved
- Include specific metrics (accuracy, F1, etc.) when available"""

COMPARISON_SYSTEM_PROMPT = """You are an academic research assistant comparing findings across multiple research papers.

COMPARISON RULES:
1. Create a structured comparison of findings across papers
2. Note AGREEMENTS: Where papers reach similar conclusions
3. Note DISAGREEMENTS: Where papers report conflicting results
4. Note DIFFERENCES in methodology, datasets, or evaluation metrics
5. Use exact numbers and citations for each paper

FORMAT:
- Use a clear structure (e.g., by metric, by method, or by finding)
- Always cite: (Author et al., Year, Section)
- Highlight key differences with specific numbers
- Provide a brief synthesis at the end"""

CONFLICT_DETECTION_PROMPT = """You are an academic research assistant identifying conflicting results across papers.

Analyze the provided research paper excerpts and:
1. Identify any CONFLICTING claims, results, or conclusions
2. For each conflict, cite both papers with their specific claims
3. Suggest possible reasons for the conflict (different datasets, methods, etc.)
4. Rate the severity of each conflict (minor/moderate/major)

Format each conflict as:
- CONFLICT: [description]
  - Paper A: [claim] (Author, Year)
  - Paper B: [claim] (Author, Year)
  - Possible reason: [explanation]
  - Severity: [minor/moderate/major]"""


def _normalize_score(score: float) -> float:
    """
    Map a retrieval score from any stage onto 0-1.

    Scores already inside [0, 1] (vector similarity, RRF fusion) pass through;
    anything else is treated as a logit and squashed. Keeps the confidence
    estimate meaningful whichever retrieval stage ran last.
    """
    if 0.0 <= score <= 1.0:
        return score
    try:
        return 1.0 / (1.0 + math.exp(-score))
    except OverflowError:
        return 0.0 if score < 0 else 1.0


class AnswerGenerator:
    """
    Generate answers with academic citations from retrieved chunks.
    
    Features:
    - Academic-aware prompts
    - Context building from RetrievalResult with paper metadata
    - Multi-paper comparison
    - Conflict detection
    - Configurable temperature and token limits
    
    Usage:
        generator = AnswerGenerator()
        results = [...]  # RetrievalResult objects from retrieval pipeline
        answer = generator.generate_answer("What accuracy did BERT achieve?", results)
        print(answer.answer)
        print(answer.papers_cited)
    """
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        temperature: Optional[float] = None,
        max_context_tokens: Optional[int] = None,
        max_answer_tokens: Optional[int] = None,
        include_citations: bool = True
    ):
        """
        Initialize AnswerGenerator.
        
        Args:
            ollama_client: OllamaClient instance (creates new if None)
            temperature: Generation temperature (0.0 = deterministic)
            max_context_tokens: Max tokens for context window
            max_answer_tokens: Max tokens for generated answer
            include_citations: Whether to request citations in answers
        """
        self.client = ollama_client or OllamaClient()
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_context_tokens = max_context_tokens or settings.max_context_tokens
        self.max_answer_tokens = max_answer_tokens or settings.answer_max_tokens
        self.include_citations = include_citations if include_citations is not None else settings.include_citations
        
        logger.info("📝 AnswerGenerator initialized")
        logger.info(f"  Max context: {self.max_context_tokens} tokens")
        logger.info(f"  Max answer: {self.max_answer_tokens} tokens")
        logger.info(f"  Citations: {self.include_citations}")
    
    def generate_answer_stream(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        context: Optional[str] = None
    ):
        """
        Generate an answer incrementally, yielding text as the model produces it.

        Uses exactly the same context and prompt construction as
        generate_answer, so a streamed answer is identical to a buffered one.

        Args:
            query: User's question
            retrieval_results: Chunks from the retrieval pipeline
            context: Optional additional context (e.g. conversation history)

        Yields:
            ("token", str) for each text fragment, then exactly one
            ("done", GeneratedAnswer) carrying sources, citations, and metrics.
            Emits ("error", str) instead of raising, so a partially streamed
            response can still be closed cleanly by the caller.
        """
        start_time = time.time()

        logger.info(f"📝 Streaming answer for: '{query}'")

        context_text, sources = self._build_context(retrieval_results)
        papers_cited = self._extract_unique_papers(sources)
        prompt = self._build_prompt(query, context_text, context)

        parts: List[str] = []
        error: Optional[str] = None

        try:
            for piece in self.client.generate_stream(
                prompt=prompt,
                system_prompt=ACADEMIC_SYSTEM_PROMPT,
                temperature=self.temperature,
            ):
                if piece:
                    parts.append(piece)
                    yield ("token", piece)
        except Exception as e:
            logger.error(f"❌ Streaming generation failed: {e}")
            error = str(e)
            yield (
                "error",
                f"Answer generation failed: {e}. "
                f"Please ensure Ollama is running with the {self.client.model} model.",
            )

        answer_text = "".join(parts)
        # Validate after the fact: tokens were already streamed, so the final
        # payload carries the cleaned text and the UI re-renders from it.
        answer_text, _removed = self._strip_unsupported_citations(answer_text, sources)
        elapsed = time.time() - start_time

        yield ("done", GeneratedAnswer(
            query=query,
            answer=answer_text,
            sources=sources,
            num_sources=len(sources),
            papers_cited=papers_cited,
            confidence=(
                0.0 if error else self._estimate_confidence(retrieval_results, answer_text)
            ),
            has_citations=self._check_has_citations(answer_text),
            is_multi_paper=len(papers_cited) > 1,
            generation_time=elapsed,
            model_used=self.client.model,
        ))

        logger.success(f"✅ Streamed {len(answer_text)} chars in {elapsed:.2f}s")

    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        context: Optional[str] = None
    ) -> GeneratedAnswer:
        """
        Generate an answer with citations from retrieved chunks.
        
        Args:
            query: User's question
            retrieval_results: List of RetrievalResult from retrieval pipeline
            context: Optional additional context (e.g., conversation history)
            
        Returns:
            GeneratedAnswer with answer text, citations, and metadata
        """
        start_time = time.time()
        
        logger.info(f"📝 Generating answer for: '{query}'")
        logger.info(f"  Sources: {len(retrieval_results)} retrieved chunks")
        
        # Build context from retrieval results
        context_text, sources = self._build_context(retrieval_results)
        
        # Determine which papers are involved
        papers_cited = self._extract_unique_papers(sources)
        is_multi_paper = len(papers_cited) > 1
        
        logger.info(f"  Papers involved: {len(papers_cited)}")
        
        # Build the prompt
        prompt = self._build_prompt(query, context_text, context)
        
        # Select system prompt
        system_prompt = ACADEMIC_SYSTEM_PROMPT
        
        # Generate answer
        try:
            answer_text = self.client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_answer_tokens
            )
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            answer_text = (
                f"I was unable to generate an answer due to a technical issue: {e}. "
                f"Please ensure Ollama is running with the {self.client.model} model."
            )
        
        # Drop any citation the retrieved sources do not support.
        answer_text, _removed = self._strip_unsupported_citations(answer_text, sources)

        elapsed = time.time() - start_time

        # Build result
        result = GeneratedAnswer(
            query=query,
            answer=answer_text,
            sources=sources,
            num_sources=len(sources),
            papers_cited=papers_cited,
            confidence=self._estimate_confidence(retrieval_results, answer_text),
            has_citations=self._check_has_citations(answer_text),
            is_multi_paper=is_multi_paper,
            generation_time=elapsed,
            model_used=self.client.model
        )
        
        logger.success(
            f"✅ Answer generated in {elapsed:.2f}s "
            f"({len(answer_text)} chars, {result.num_sources} sources, "
            f"{len(papers_cited)} papers)"
        )
        
        return result
    
    def compare_papers(
        self,
        query: str,
        retrieval_results: List[RetrievalResult]
    ) -> GeneratedAnswer:
        """
        Generate a multi-paper comparison for the given query.
        
        Groups results by paper and generates a structured comparison.
        
        Args:
            query: The comparison question
            retrieval_results: Retrieved chunks from multiple papers
            
        Returns:
            GeneratedAnswer with comparison analysis
        """
        start_time = time.time()
        
        logger.info(f"📊 Generating paper comparison for: '{query}'")
        
        # Build context grouped by paper
        context_text, sources = self._build_context_by_paper(retrieval_results)
        papers_cited = self._extract_unique_papers(sources)
        
        if len(papers_cited) < 2:
            logger.warning("⚠️ Less than 2 papers found for comparison")
        
        # Build comparison prompt
        prompt = self._build_comparison_prompt(query, context_text)
        
        try:
            answer_text = self.client.generate(
                prompt=prompt,
                system_prompt=COMPARISON_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_answer_tokens
            )
        except Exception as e:
            logger.error(f"❌ Comparison generation failed: {e}")
            answer_text = f"Unable to generate comparison: {e}"
        
        elapsed = time.time() - start_time
        
        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            sources=sources,
            num_sources=len(sources),
            papers_cited=papers_cited,
            confidence=self._estimate_confidence(retrieval_results, answer_text),
            has_citations=self._check_has_citations(answer_text),
            is_multi_paper=True,
            generation_time=elapsed,
            model_used=self.client.model
        )
    
    def detect_conflicts(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> GeneratedAnswer:
        """
        Detect conflicting results across multiple papers.
        
        Args:
            retrieval_results: Retrieved chunks from multiple papers
            
        Returns:
            GeneratedAnswer describing any conflicts found
        """
        start_time = time.time()
        
        logger.info("🔍 Detecting conflicts across papers")
        
        context_text, sources = self._build_context_by_paper(retrieval_results)
        papers_cited = self._extract_unique_papers(sources)
        
        prompt = (
            f"Analyze the following research paper excerpts and identify "
            f"any conflicting claims, results, or conclusions.\n\n"
            f"{context_text}\n\n"
            f"Identify all conflicts between these papers:"
        )
        
        try:
            answer_text = self.client.generate(
                prompt=prompt,
                system_prompt=CONFLICT_DETECTION_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_answer_tokens
            )
        except Exception as e:
            logger.error(f"❌ Conflict detection failed: {e}")
            answer_text = f"Unable to detect conflicts: {e}"
        
        elapsed = time.time() - start_time
        
        return GeneratedAnswer(
            query="Detect conflicting results across papers",
            answer=answer_text,
            sources=sources,
            num_sources=len(sources),
            papers_cited=papers_cited,
            confidence=0.5,  # Conflict detection is inherently uncertain
            has_citations=self._check_has_citations(answer_text),
            is_multi_paper=len(papers_cited) > 1,
            generation_time=elapsed,
            model_used=self.client.model
        )
    
    # ==================================================================
    # CONTEXT BUILDING
    # ==================================================================
    
    def _build_context(
        self,
        results: List[RetrievalResult]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Build context string from retrieval results with paper metadata.
        
        Each source is formatted with its paper metadata for the LLM to cite.
        
        Args:
            results: List of RetrievalResult
            
        Returns:
            Tuple of (context_text, sources_list)
        """
        context_parts = []
        sources = []
        
        for i, result in enumerate(results, 1):
            # Extract paper metadata from chunk
            metadata = result.metadata or {}
            
            paper_title = metadata.get("paper_title", "Unknown Document")
            authors = metadata.get("authors", [])
            year = metadata.get("year", "n.d.")
            venue = metadata.get("venue", "")
            section_type = metadata.get("section_type", "unknown")
            
            # Format author string for citations
            author_str = self._format_author_string(authors)
            
            # Build source header
            header = f"[Source {i}]"
            if paper_title != "Unknown Document":
                header += f" Paper: {paper_title}"
            if author_str:
                header += f" | Authors: {author_str}"
            if year != "n.d.":
                header += f" | Year: {year}"
            if venue:
                header += f" | Venue: {venue}"
            if section_type != "unknown":
                header += f" | Section: {section_type}"

            # Small local models reliably copy a literal string but often fail
            # to assemble a citation from separate metadata fields, so hand
            # them the exact text to reproduce.
            cite_as = self._build_citation_string(author_str, year, section_type)
            if cite_as:
                header += f"\nCite this source as: {cite_as}"

            # Add the chunk text
            source_block = f"{header}\n{result.text}\n"
            context_parts.append(source_block)
            
            # Track source metadata
            sources.append({
                "source_num": i,
                "paper_title": paper_title,
                "authors": authors,
                "year": year,
                "venue": venue,
                "section_type": section_type,
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "score": result.score,
                "text_preview": result.text[:200]
            })
        
        context_text = "\n---\n".join(context_parts)
        return context_text, sources
    
    def _build_context_by_paper(
        self,
        results: List[RetrievalResult]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Build context grouped by paper for comparison queries.
        
        Args:
            results: List of RetrievalResult
            
        Returns:
            Tuple of (context_text, sources_list)
        """
        # Group results by paper title
        paper_groups: Dict[str, List[RetrievalResult]] = {}
        for result in results:
            title = (result.metadata or {}).get("paper_title", "Unknown Document")
            if title not in paper_groups:
                paper_groups[title] = []
            paper_groups[title].append(result)
        
        context_parts = []
        sources = []
        source_num = 1
        
        for paper_title, paper_results in paper_groups.items():
            # Paper header
            first = paper_results[0]
            metadata = first.metadata or {}
            authors = metadata.get("authors", [])
            year = metadata.get("year", "n.d.")
            venue = metadata.get("venue", "")
            author_str = self._format_author_string(authors)
            
            paper_header = f"=== PAPER: {paper_title} ==="
            if author_str:
                paper_header += f"\nAuthors: {author_str}"
            if year != "n.d.":
                paper_header += f" | Year: {year}"
            if venue:
                paper_header += f" | Venue: {venue}"
            
            # Add all chunks from this paper
            chunks = []
            for result in paper_results:
                section = (result.metadata or {}).get("section_type", "unknown")
                chunks.append(f"[{section}] {result.text}")
                
                sources.append({
                    "source_num": source_num,
                    "paper_title": paper_title,
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "section_type": section,
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "score": result.score,
                    "text_preview": result.text[:200]
                })
                source_num += 1
            
            paper_block = f"{paper_header}\n\n" + "\n\n".join(chunks)
            context_parts.append(paper_block)
        
        context_text = "\n\n" + "=" * 50 + "\n\n".join(context_parts)
        return context_text, sources
    
    # ==================================================================
    # PROMPT BUILDING
    # ==================================================================
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Build the generation prompt with context and question."""
        parts = []
        
        parts.append("Based on the following source documents, answer the question.")
        parts.append("Cite your sources using the format: (Author et al., Year, Section)")
        parts.append("")
        parts.append("SOURCE DOCUMENTS:")
        parts.append(context)
        parts.append("")
        
        if additional_context:
            parts.append(f"ADDITIONAL CONTEXT:\n{additional_context}\n")
        
        parts.append(f"QUESTION: {query}")
        parts.append("")
        parts.append("ANSWER (with citations):")
        
        return "\n".join(parts)
    
    def _build_comparison_prompt(self, query: str, context: str) -> str:
        """Build prompt for multi-paper comparison."""
        return (
            f"Compare the following research papers to answer this question:\n\n"
            f"QUESTION: {query}\n\n"
            f"PAPER EXCERPTS:\n{context}\n\n"
            f"Provide a structured comparison with citations:"
        )
    
    # ==================================================================
    # UTILITY METHODS
    # ==================================================================
    
    # Parenthetical citation: "(Saadi et al., 2025, methodology)". Opens with a
    # capitalized name and contains a year. Narrow, so ordinary parentheses
    # such as "(a CNN)" or "(roughly two days)" are left alone.
    _CITATION_RE = re.compile(r'\(\s*[A-Z][^()]{0,120}?\b(?:\d{4}|n\.d\.)[^()]{0,60}?\)')

    # Narrative citation: "Saadi et al. (2025, methodology)" — the name sits
    # outside the parentheses, so the pattern above never sees it.
    _NARRATIVE_CITATION_RE = re.compile(
        # Optional attribution lead-in, absorbed so removing the citation does
        # not leave "According to, ..." stranded at the front of a sentence.
        r'(?:\b(?:According to|As reported (?:by|in)|As shown (?:by|in)|'
        r'As described (?:by|in)|Per|Following)\s+)?'
        r'\b(?P<surname>[A-Z][A-Za-z\-\']+)'                  # surname
        r'(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z\-\']+))?'    # "et al." / "and Abghour"
        r'\s*\(\s*(?P<year>\d{4}|n\.d\.)[^()]{0,60}?\)'       # (year[, section])
        r',?\s*'                                              # trailing comma
    )

    @staticmethod
    def _surname(name: str) -> str:
        """Last token of a name, lowercased — how citations refer to authors."""
        parts = (name or "").split()
        return parts[-1].lower() if parts else ""

    def _supported_author_years(self, sources: List[Dict[str, Any]]) -> set:
        """
        (surname, year) pairs the retrieved sources actually support.

        Matching on surname and year rather than the whole citation string is
        deliberate: the model may render the same source as "Saadi et al.",
        "Saadi and Abghour", or "A. Saadi", and all three are legitimate.
        Author identity plus year is the part that must be true.
        """
        pairs = set()
        for s in sources:
            year = str(s.get("year") or "").strip()
            if not year or year in {"0", "None"}:
                continue
            for author in (s.get("authors") or []):
                surname = self._surname(author)
                if surname:
                    pairs.add((surname, year))
        return pairs

    def _strip_unsupported_citations(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> tuple[str, int]:
        """
        Remove citations that no retrieved source supports.

        Small models invent plausible-looking citations even when the prompt
        forbids it, and a fabricated citation is worse than none in a system
        whose purpose is grounded attribution. Prompting cannot guarantee this,
        so unsupported citations are deleted rather than trusted.

        Handles both renderings the model produces:
            parenthetical  "(Saadi et al., 2025, methodology)"
            narrative      "According to Saadi et al. (2025), ..."

        Returns:
            (cleaned answer, number of citations removed)
        """
        supported = self._supported_author_years(sources)
        removed = 0

        def is_supported(surname: str, year: str) -> bool:
            return (surname.lower(), year) in supported

        def replace_parenthetical(match: "re.Match") -> str:
            nonlocal removed
            text = match.group(0)
            year_match = re.search(r'\b(\d{4})\b', text)
            name_match = re.search(r'\(\s*([A-Z][A-Za-z\-\']+)', text)
            if year_match and name_match and is_supported(name_match.group(1), year_match.group(1)):
                return text
            removed += 1
            logger.warning(f"Removed unsupported citation: {text}")
            return ""

        cleaned = self._CITATION_RE.sub(replace_parenthetical, answer)

        def replace_narrative(match: "re.Match") -> str:
            nonlocal removed
            if is_supported(match.group("surname"), match.group("year")):
                return match.group(0)
            removed += 1
            logger.warning(f"Removed unsupported citation: {match.group(0)}")
            # Drop the lead-in too ("According to X (2019),"), which would
            # otherwise be left dangling in front of the sentence.
            return ""

        cleaned = self._NARRATIVE_CITATION_RE.sub(replace_narrative, cleaned)

        if removed:
            cleaned = self._tidy_after_removal(cleaned)

        return cleaned, removed

    @staticmethod
    def _tidy_after_removal(text: str) -> str:
        """Repair spacing and punctuation left behind by a deleted citation."""
        text = re.sub(r"\(\s*\)", "", text)                    # empty parens
        text = re.sub(r"[ \t]{2,}", " ", text)                 # doubled spaces
        text = re.sub(r"\s+([.,;:])", r"\1", text)             # space before punctuation
        text = re.sub(r"([.,;:])(?:\s*\1)+", r"\1", text)      # doubled punctuation
        text = re.sub(r"(^|[.!?]\s+)([a-z])",                  # re-capitalize a sentence
                      lambda m: m.group(1) + m.group(2).upper(), text)
        return text.strip()

    @staticmethod
    def _build_citation_string(author_str: str, year, section_type: str) -> str:
        """
        Build the exact inline citation for a source, e.g. "(Saadi et al., 2025, results)".

        Returns an empty string when there is no attribution to cite.
        """
        if not author_str and (year in (None, "", "n.d.")):
            return ""

        parts = [author_str] if author_str else []
        parts.append(str(year) if year not in (None, "", "n.d.") else "n.d.")
        if section_type and section_type != "unknown":
            parts.append(section_type)
        return f"({', '.join(parts)})"

    @staticmethod
    def _format_author_string(authors: List[str]) -> str:
        """Format author list for citations (e.g., 'Devlin et al.')."""
        if not authors:
            return ""
        if len(authors) == 1:
            return authors[0]
        if len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        # 3+ authors: "First et al."
        last_name = authors[0].split()[-1] if authors[0] else "Unknown"
        return f"{last_name} et al."
    
    @staticmethod
    def _extract_unique_papers(sources: List[Dict[str, Any]]) -> List[str]:
        """Extract unique paper titles from sources."""
        seen = set()
        papers = []
        for source in sources:
            title = source.get("paper_title", "Unknown Document")
            if title not in seen and title != "Unknown Document":
                seen.add(title)
                papers.append(title)
        return papers
    
    @staticmethod
    def _estimate_confidence(
        results: List[RetrievalResult],
        answer_text: str
    ) -> float:
        """
        Estimate confidence in the generated answer.
        
        Based on:
        - Retrieval scores (higher = better evidence)
        - Number of sources (more = better supported)
        - Answer length (too short may indicate insufficient context)
        """
        if not results or not answer_text:
            return 0.0

        # Average retrieval score, normalized to 0-1 first.
        #
        # `score` carries whichever stage ran last, and the stages are on
        # different scales: vector similarity is already 0-1, RRF fusion is a
        # small positive number, but a cross-encoder emits an unbounded logit
        # that is routinely negative (roughly -11..+11 for ms-marco). Averaging
        # those raw made the weighted sum negative, and the clamp below turned
        # every reranked answer's confidence into exactly 0.0. A logistic maps
        # any of them onto a comparable 0-1 range.
        avg_score = sum(_normalize_score(r.score) for r in results) / len(results)

        # Source count factor (1-5 sources, diminishing returns)
        source_factor = min(len(results) / 3.0, 1.0)
        
        # Answer quality factor (reasonable length)
        answer_len = len(answer_text)
        if answer_len < 50:
            quality_factor = 0.3
        elif answer_len < 200:
            quality_factor = 0.7
        else:
            quality_factor = 1.0
        
        confidence = (avg_score * 0.5 + source_factor * 0.3 + quality_factor * 0.2)
        return min(max(confidence, 0.0), 1.0)
    
    @staticmethod
    def _check_has_citations(answer_text: str) -> bool:
        """
        Whether the answer carries at least one citation.

        Recognizes both the parenthetical form, "(Devlin et al., 2018)", and
        the narrative form, "Devlin et al. (2018)", which models produce just
        as often. Runs after unsupported citations are stripped, so a True
        result means a source-backed citation survived.
        """
        return bool(
            AnswerGenerator._CITATION_RE.search(answer_text)
            or AnswerGenerator._NARRATIVE_CITATION_RE.search(answer_text)
        )


# Convenience function
def generate_answer(
    query: str,
    retrieval_results: List[RetrievalResult],
    **kwargs
) -> GeneratedAnswer:
    """
    Quick answer generation with default settings.
    
    Args:
        query: User question
        retrieval_results: Retrieved chunks
        **kwargs: Additional AnswerGenerator parameters
        
    Returns:
        GeneratedAnswer with citations
    """
    generator = AnswerGenerator(**kwargs)
    return generator.generate_answer(query, retrieval_results)
