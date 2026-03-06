"""
Citation Formatter - Academic citation formatting for research papers.

This is Step 3 of Phase 4: Answer Generation with Citations.

PURPOSE:
--------
Format academic citations in multiple styles:
1. Inline citations: (Devlin et al., 2018, Results, p.9)
2. Reference list: Numbered bibliography entries
3. BibTeX: Standard LaTeX bibliography format

WHY THIS MATTERS:
-----------------
Proper academic citations are what differentiate this system from
generic RAG chatbots. Every claim must be traceable to a specific
paper, section, and ideally page number.

CITATION STYLES:
-----------------
1. **Inline (APA-like)**: (Author et al., Year, Section, p.X)
   - Used within answer text
   - Quick reference for readers
   
2. **Reference List**: 
   [1] Devlin, J., et al. (2018). BERT: Pre-training of Deep
       Bidirectional Transformers. NAACL.
   - Full bibliographic entries at end of answer
   
3. **BibTeX**:
   @article{devlin2018bert,
     title={BERT: Pre-training...},
     author={Jacob Devlin and Ming-Wei Chang},
     year={2018},
     journal={NAACL}
   }
   - For LaTeX documents and academic papers

Author: DocuSense
Created: 2026-03-06
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class CitationStyle(Enum):
    """Supported citation styles."""
    INLINE = "inline"       # (Author et al., Year, Section)
    NUMBERED = "numbered"   # [1], [2], [3]
    REFERENCE = "reference" # Full reference list
    BIBTEX = "bibtex"       # BibTeX format


@dataclass
class FormattedCitation:
    """A single formatted citation."""
    citation_num: int                  # [1], [2], etc.
    inline: str                        # (Devlin et al., 2018, Results)
    reference: str                     # Full reference entry
    bibtex: str                        # BibTeX entry
    paper_title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    section_type: Optional[str] = None
    doi: Optional[str] = None


class CitationFormatter:
    """
    Format academic citations from retrieval source metadata.
    
    Features:
    - Multiple citation styles (APA inline, numbered, reference list, BibTeX)
    - Deduplication (group chunks from same paper)
    - Smart author formatting (et al. for 3+ authors)
    - Section and page number inclusion
    
    Usage:
        formatter = CitationFormatter()
        sources = [{"paper_title": "BERT...", "authors": [...], ...}]
        citations = formatter.format_sources(sources)
        
        for c in citations:
            print(c.inline)      # (Devlin et al., 2018, Results)
            print(c.reference)   # [1] Devlin, J., et al. (2018)...
            print(c.bibtex)      # @article{devlin2018bert, ...}
    """
    
    def format_sources(
        self,
        sources: List[Dict[str, Any]],
        style: CitationStyle = CitationStyle.INLINE
    ) -> List[FormattedCitation]:
        """
        Format a list of source metadata into citations.
        
        Deduplicates by paper title - chunks from the same paper
        are merged into a single citation.
        
        Args:
            sources: List of source metadata dicts (from AnswerGenerator)
            style: Citation style to use
            
        Returns:
            List of FormattedCitation objects (one per unique paper)
        """
        logger.info(f"📎 Formatting {len(sources)} sources as {style.value} citations")
        
        # Deduplicate by paper title
        unique_papers = self._deduplicate_sources(sources)
        
        citations = []
        for i, (title, paper_data) in enumerate(unique_papers.items(), 1):
            citation = self._format_single_citation(i, paper_data)
            citations.append(citation)
        
        logger.info(f"📎 Created {len(citations)} unique citations from {len(sources)} sources")
        return citations
    
    def format_reference_list(
        self,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a formatted reference list string.
        
        Args:
            sources: List of source metadata dicts
            
        Returns:
            Formatted reference list as a single string
        """
        citations = self.format_sources(sources)
        
        if not citations:
            return ""
        
        lines = ["", "References:", ""]
        for citation in citations:
            lines.append(citation.reference)
        
        return "\n".join(lines)
    
    def format_bibtex_export(
        self,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Generate BibTeX entries for all cited papers.
        
        Args:
            sources: List of source metadata dicts
            
        Returns:
            BibTeX string with all entries
        """
        citations = self.format_sources(sources)
        
        if not citations:
            return ""
        
        entries = [citation.bibtex for citation in citations]
        return "\n\n".join(entries)
    
    def format_inline_citation(
        self,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Format a single inline citation from metadata.
        
        Args:
            metadata: Source metadata dict
            
        Returns:
            Inline citation string, e.g., (Devlin et al., 2018, Results, p.9)
        """
        authors = metadata.get("authors", [])
        year = metadata.get("year", "n.d.")
        section = metadata.get("section_type", "")
        page = metadata.get("page", "")
        
        # Format author
        author_str = self._format_author_short(authors)
        
        # Build citation parts
        parts = []
        if author_str:
            parts.append(author_str)
        parts.append(str(year))
        if section and section != "unknown":
            parts.append(section.capitalize())
        if page:
            parts.append(f"p.{page}")
        
        return f"({', '.join(parts)})"
    
    # ==================================================================
    # INTERNAL METHODS
    # ==================================================================
    
    def _deduplicate_sources(
        self,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Deduplicate sources by paper title, merging section info.
        
        Returns:
            Dict mapping paper_title -> merged metadata
        """
        papers: Dict[str, Dict[str, Any]] = {}
        
        for source in sources:
            title = source.get("paper_title", "Unknown Document")
            
            if title not in papers:
                papers[title] = {
                    "paper_title": title,
                    "authors": source.get("authors", []),
                    "year": source.get("year"),
                    "venue": source.get("venue", ""),
                    "doi": source.get("doi", ""),
                    "sections": set(),
                    "chunks": []
                }
            
            # Add section type
            section = source.get("section_type", "unknown")
            if section and section != "unknown":
                papers[title]["sections"].add(section)
            
            # Track chunks from this paper
            papers[title]["chunks"].append(source)
        
        return papers
    
    def _format_single_citation(
        self,
        num: int,
        paper_data: Dict[str, Any]
    ) -> FormattedCitation:
        """Format a single citation in all styles."""
        title = paper_data["paper_title"]
        authors = paper_data.get("authors", [])
        year = paper_data.get("year")
        venue = paper_data.get("venue", "")
        doi = paper_data.get("doi", "")
        sections = paper_data.get("sections", set())
        
        # Inline citation
        inline = self._make_inline(authors, year, sections)
        
        # Full reference
        reference = self._make_reference(num, title, authors, year, venue, doi)
        
        # BibTeX
        bibtex = self._make_bibtex(title, authors, year, venue, doi)
        
        return FormattedCitation(
            citation_num=num,
            inline=inline,
            reference=reference,
            bibtex=bibtex,
            paper_title=title,
            authors=authors,
            year=year,
            venue=venue,
            section_type=", ".join(sorted(sections)) if sections else None,
            doi=doi if doi else None
        )
    
    def _make_inline(
        self,
        authors: List[str],
        year: Optional[int],
        sections: set
    ) -> str:
        """Create inline citation: (Devlin et al., 2018, Results)"""
        parts = []
        
        author_str = self._format_author_short(authors)
        if author_str:
            parts.append(author_str)
        
        parts.append(str(year) if year else "n.d.")
        
        if sections:
            # Include first section for inline
            section_str = ", ".join(s.capitalize() for s in sorted(sections)[:2])
            parts.append(section_str)
        
        return f"({', '.join(parts)})"
    
    def _make_reference(
        self,
        num: int,
        title: str,
        authors: List[str],
        year: Optional[int],
        venue: str,
        doi: str
    ) -> str:
        """
        Create full reference entry:
        [1] Devlin, J., et al. (2018). BERT: Pre-training of Deep
            Bidirectional Transformers. NAACL. DOI: 10.18653/v1/N19-1423
        """
        parts = [f"[{num}]"]
        
        # Authors
        if authors:
            author_refs = []
            for author in authors[:3]:  # Max 3 authors in reference
                names = author.split()
                if len(names) >= 2:
                    # Last, F.
                    author_refs.append(f"{names[-1]}, {names[0][0]}.")
                else:
                    author_refs.append(author)
            
            author_str = ", ".join(author_refs)
            if len(authors) > 3:
                author_str += ", et al."
            parts.append(author_str)
        
        # Year
        year_str = f"({year})" if year else "(n.d.)"
        parts.append(year_str + ".")
        
        # Title
        parts.append(f"{title}.")
        
        # Venue
        if venue:
            parts.append(f"{venue}.")
        
        # DOI
        if doi:
            parts.append(f"DOI: {doi}")
        
        return " ".join(parts)
    
    def _make_bibtex(
        self,
        title: str,
        authors: List[str],
        year: Optional[int],
        venue: str,
        doi: str
    ) -> str:
        """
        Create BibTeX entry:
        @article{key,
          title={...},
          author={...},
          year={...},
          journal={...}
        }
        """
        # Generate key: first_author_lastname + year + first_word_of_title
        key = self._generate_bibtex_key(authors, year, title)
        
        # Determine entry type
        entry_type = "inproceedings" if venue else "article"
        
        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title={{{title}}},")
        
        if authors:
            author_str = " and ".join(authors)
            lines.append(f"  author={{{author_str}}},")
        
        if year:
            lines.append(f"  year={{{year}}},")
        
        if venue:
            field_name = "booktitle" if entry_type == "inproceedings" else "journal"
            lines.append(f"  {field_name}={{{venue}}},")
        
        if doi:
            lines.append(f"  doi={{{doi}}},")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_author_short(authors: List[str]) -> str:
        """Format authors for inline citation (et al. for 3+)."""
        if not authors:
            return ""
        
        # Get last name of first author
        first = authors[0]
        names = first.split()
        last_name = names[-1] if names else first
        
        if len(authors) == 1:
            return last_name
        elif len(authors) == 2:
            second = authors[1].split()
            second_last = second[-1] if second else authors[1]
            return f"{last_name} and {second_last}"
        else:
            return f"{last_name} et al."
    
    @staticmethod
    def _generate_bibtex_key(
        authors: List[str],
        year: Optional[int],
        title: str
    ) -> str:
        """Generate a BibTeX citation key."""
        # First author last name (lowercase)
        if authors:
            first_author = authors[0].split()[-1].lower()
        else:
            first_author = "unknown"
        
        # Year
        year_str = str(year) if year else "nd"
        
        # First significant word of title (lowercase, skip common words)
        skip_words = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with"}
        title_words = re.sub(r'[^\w\s]', '', title.lower()).split()
        first_word = "paper"
        for word in title_words:
            if word not in skip_words:
                first_word = word
                break
        
        return f"{first_author}{year_str}{first_word}"


# Convenience function
def format_citation(
    metadata: Dict[str, Any],
    style: CitationStyle = CitationStyle.INLINE
) -> str:
    """
    Quick citation formatting.
    
    Args:
        metadata: Source metadata dict
        style: Citation style
        
    Returns:
        Formatted citation string
    """
    formatter = CitationFormatter()
    return formatter.format_inline_citation(metadata)
