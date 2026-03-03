"""
Research Paper Metadata Extractor - Academic document analysis.

This module extracts research paper specific metadata:
- Title, Authors, Affiliations
- Abstract, Keywords
- Publication info (year, venue, DOI)
- Section structure (Introduction, Methodology, Results, etc.)
- Equations, Citations, Figures, Tables

PURPOSE:
--------
Transform generic RAG into a RESEARCH PAPER ANALYSIS SYSTEM by:
1. Identifying paper structure (IEEE, ACL, NeurIPS, arXiv layouts)
2. Extracting bibliographic metadata
3. Detecting academic sections
4. Preserving citations and references
5. Enabling section-specific retrieval (e.g., "show only Results")

WHY THIS MATTERS:
-----------------
- General RAG: "The document mentions 93% accuracy"
- Research RAG: "BERT achieved 93.5% F1 on SST-2 (Devlin et al., 2018, Table 4, p.9)"

DETECTION STRATEGIES:
---------------------
1. **Title**: Large font on first page, typically ALL CAPS or Title Case
2. **Authors**: Below title, often with superscript numbers for affiliations
3. **Abstract**: Section labeled "Abstract" or "ABSTRACT" on first page
4. **Sections**: Numbered headers (1. Introduction, 2. Related Work)
5. **References**: Section labeled "References" or "Bibliography" at end
6. **DOI**: Pattern "10.XXXX/..." usually on first page
7. **Year**: 4-digit number near title or in footer

Author: DocuSense
Created: 2026-03-03
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
import PyPDF2


@dataclass
class PaperSection:
    """Represents a section in a research paper."""
    section_type: str  # "abstract", "introduction", "methodology", "results", "discussion", "conclusion", "references"
    title: str  # "3.2 Experimental Setup"
    level: int  # 1=major section, 2=subsection, 3=subsubsection
    start_page: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    has_equations: bool = False
    has_tables: bool = False
    has_figures: bool = False


@dataclass
class Citation:
    """Represents an in-text citation."""
    text: str  # "[23]" or "(Smith et al., 2020)"
    position: int  # Character position in document
    citation_type: str  # "numbered" or "author-year"


@dataclass
class PaperMetadata:
    """
    Complete metadata for a research paper.
    
    This is what makes your RAG system ACADEMIC-GRADE!
    """
    # Basic info
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    
    # Publication info
    year: Optional[int] = None
    venue: Optional[str] = None  # Conference or journal name
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    
    # Content sections
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    sections: List[PaperSection] = field(default_factory=list)
    
    # References
    num_references: int = 0
    citations: List[Citation] = field(default_factory=list)
    
    # Document properties
    total_pages: int = 0
    paper_type: str = "unknown"  # "conference", "journal", "arxiv", "thesis"
    
    # Detection confidence
    confidence: float = 0.0  # 0-1 score: how confident we are this is a research paper
    
    def is_research_paper(self) -> bool:
        """Check if this looks like a research paper (vs. generic document)."""
        return self.confidence > 0.5
    
    def get_section_type(self, char_position: int) -> str:
        """Get section type for a given character position."""
        for section in self.sections:
            if section.start_char and section.end_char:
                if section.start_char <= char_position < section.end_char:
                    return section.section_type
        return "unknown"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "title": self.title,
            "authors": self.authors,
            "affiliations": self.affiliations,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "num_references": self.num_references,
            "total_pages": self.total_pages,
            "paper_type": self.paper_type,
            "confidence": self.confidence
        }


class PaperMetadataExtractor:
    """
    Extracts research paper metadata from PDF/Markdown.
    
    WORKFLOW:
    ---------
    1. Detect if document is a research paper (vs. book, report, etc.)
    2. Extract bibliographic info (title, authors, year)
    3. Find abstract and keywords
    4. Identify section structure
    5. Extract citations and references
    6. Calculate confidence score
    
    DETECTION HEURISTICS:
    ---------------------
    High confidence if document has:
    - Abstract section on first page
    - Numbered references section
    - Author list with affiliations
    - Academic keywords (methodology, experimental, evaluation)
    - Citation patterns ([1], (Author, 2020))
    """
    
    # Common section headers in research papers
    SECTION_PATTERNS = {
        "abstract": r"\b(abstract|summary)\b",
        "introduction": r"\b(introduction|background)\b",
        "related_work": r"\b(related work|literature review|prior work|previous work)\b",
        "methodology": r"\b(method(ology)?|approach|model|architecture|system)\b",
        "experiments": r"\b(experiment(s|al)?|evaluation|validation)\b",
        "results": r"\b(results?|findings|performance)\b",
        "discussion": r"\b(discussion|analysis)\b",
        "conclusion": r"\b(conclusion(s)?|summary|future work)\b",
        "references": r"\b(references|bibliography)\b"
    }
    
    # Common academic venues (for detection)
    VENUES = [
        "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV", "ACL", "EMNLP",
        "NAACL", "AAAI", "IJCAI", "KDD", "WWW", "SIGIR", "ICSE", "FSE",
        "IEEE", "ACM", "Springer", "Nature", "Science", "arXiv"
    ]
    
    def __init__(self):
        """Initialize the metadata extractor."""
        logger.info("PaperMetadataExtractor initialized")
    
    def extract_from_markdown(self, markdown: str, file_path: Optional[Path] = None) -> PaperMetadata:
        """
        Extract paper metadata from Markdown text.
        
        Args:
            markdown: Converted Markdown text from PDF
            file_path: Optional original PDF path for additional extraction
            
        Returns:
            PaperMetadata with extracted information
        """
        logger.info("Extracting research paper metadata")
        
        metadata = PaperMetadata()
        
        # Extract components
        metadata.title = self._extract_title(markdown)
        metadata.authors = self._extract_authors(markdown)
        metadata.year = self._extract_year(markdown)
        metadata.venue = self._extract_venue(markdown)
        metadata.doi = self._extract_doi(markdown)
        metadata.arxiv_id = self._extract_arxiv_id(markdown)
        metadata.abstract = self._extract_abstract(markdown)
        metadata.keywords = self._extract_keywords(markdown)
        metadata.sections = self._extract_sections(markdown)
        metadata.citations = self._extract_citations(markdown)
        metadata.num_references = self._count_references(markdown)
        
        # Extract from PDF if available
        if file_path and file_path.suffix.lower() == '.pdf':
            pdf_metadata = self._extract_from_pdf(file_path)
            metadata = self._merge_metadata(metadata, pdf_metadata)
        
        # Calculate confidence
        metadata.confidence = self._calculate_confidence(metadata)
        
        # Determine paper type
        metadata.paper_type = self._determine_paper_type(metadata)
        
        logger.info(f"Metadata extraction complete (confidence: {metadata.confidence:.2f})")
        logger.info(f"  Title: {metadata.title[:50] if metadata.title else 'Not found'}...")
        logger.info(f"  Authors: {len(metadata.authors)} detected")
        logger.info(f"  Year: {metadata.year}")
        logger.info(f"  Sections: {len(metadata.sections)} detected")
        
        return metadata
    
    def _extract_title(self, markdown: str) -> Optional[str]:
        """
        Extract paper title.
        
        Strategy:
        1. Look for # header on first ~500 chars (main title in Markdown)
        2. Look for lines in ALL CAPS or Title Case
        3. Filter out common non-title patterns (page numbers, URLs)
        """
        # Get first ~1000 characters (title should be near top)
        top_section = markdown[:1000]
        
        # Strategy 1: First # header (most reliable for Markdown)
        match = re.search(r'^#\s+(.+)$', top_section, re.MULTILINE)
        if match:
            title = match.group(1).strip()
            # Clean up (remove trailing asterisks, etc.)
            title = re.sub(r'[*_]+$', '', title).strip()
            if len(title) > 10 and len(title) < 300:  # Reasonable title length
                return title
        
        # Strategy 2: Look for lines in Title Case or ALL CAPS
        lines = top_section.split('\n')
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            # Skip short lines, URLs, page numbers
            if len(line) < 10 or len(line) > 300:
                continue
            if 'http' in line.lower() or 'www' in line.lower():
                continue
            if re.match(r'^\d+$', line):  # Just a number
                continue
            
            # Check if Title Case or ALL CAPS
            if line.isupper() or (line[0].isupper() and sum(c.isupper() for c in line) >= 3):
                # Likely a title
                return line
        
        return None
    
    def _extract_authors(self, markdown: str) -> List[str]:
        """
        Extract author names.
        
        Strategy:
        1. Look for patterns like "John Smith, Jane Doe"
        2. Look for author annotations (*, †, 1, 2)
        3. Filter out common false positives
        """
        authors = []
        
        # Get first ~2000 characters (authors should be near top)
        top_section = markdown[:2000]
        
        # Look for comma-separated names with possible annotations
        # Pattern: "First Last, First Last" or "First Last1, First Last2"
        pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+[*†‡§¶0-9]*)'
        matches = re.findall(pattern, top_section)
        
        for match in matches:
            # Clean up annotations
            name = re.sub(r'[*†‡§¶0-9]+$', '', match).strip()
            # Reasonable name length
            if 5 <= len(name) <= 50 and name.count(' ') >= 1:
                authors.append(name)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_authors = []
        for author in authors:
            if author not in seen:
                seen.add(author)
                unique_authors.append(author)
        
        return unique_authors[:20]  # Max 20 authors (reasonable limit)
    
    def _extract_year(self, markdown: str) -> Optional[int]:
        """
        Extract publication year.
        
        Strategy:
        1. Look for 4-digit years in first ~2000 chars (2000-2099)
        2. Prefer years near "copyright", "published", dates
        """
        top_section = markdown[:2000]
        
        # Find all 4-digit years (2000-2099)
        years = re.findall(r'\b(20[0-2][0-9])\b', top_section)
        
        if years:
            # Return most recent year (often the publication year)
            return int(max(years))
        
        return None
    
    def _extract_venue(self, markdown: str) -> Optional[str]:
        """Extract publication venue (conference/journal)."""
        top_section = markdown[:2000]
        
        # Check for known venue names
        for venue in self.VENUES:
            if re.search(r'\b' + re.escape(venue) + r'\b', top_section, re.IGNORECASE):
                return venue
        
        return None
    
    def _extract_doi(self, markdown: str) -> Optional[str]:
        """
        Extract DOI (Digital Object Identifier).
        
        Pattern: 10.XXXX/...
        """
        match = re.search(r'\b(10\.\d{4,}/[^\s]+)\b', markdown)
        if match:
            doi = match.group(1)
            # Clean trailing punctuation
            doi = re.sub(r'[.,;)\]]+$', '', doi)
            return doi
        return None
    
    def _extract_arxiv_id(self, markdown: str) -> Optional[str]:
        """
        Extract arXiv ID.
        
        Patterns:
        - Old: arxiv:0706.0001
        - New: arxiv:2103.12345 or arXiv:2103.12345v1
        """
        match = re.search(r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)', markdown, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _extract_abstract(self, markdown: str) -> Optional[str]:
        """
        Extract abstract text.
        
        Strategy:
        1. Look for "Abstract" header
        2. Extract text until next section header
        """
        # Find "Abstract" header (case-insensitive)
        pattern = r'#{1,3}\s*Abstract\s*\n(.*?)(?=\n#{1,3}\s|\n\n[A-Z]|\Z)'
        match = re.search(pattern, markdown, re.IGNORECASE | re.DOTALL)
        
        if match:
            abstract = match.group(1).strip()
            # Clean up extra whitespace
            abstract = re.sub(r'\s+', ' ', abstract)
            if 50 <= len(abstract) <= 3000:  # Reasonable abstract length
                return abstract
        
        # Fallback: Look for "Abstract" followed by paragraph
        pattern2 = r'\bAbstract[:\s]+(.*?)(?=\n\n|\n[A-Z][a-z]+:)'
        match2 = re.search(pattern2, markdown, re.IGNORECASE | re.DOTALL)
        if match2:
            abstract = match2.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            if 50 <= len(abstract) <= 3000:
                return abstract
        
        return None
    
    def _extract_keywords(self, markdown: str) -> List[str]:
        """
        Extract keywords/keyphrases.
        
        Look for "Keywords:", "Index Terms:", etc.
        """
        keywords = []
        
        # Pattern: Keywords: word1, word2, word3
        pattern = r'\b(?:Keywords?|Index Terms?)[:\s]+(.*?)(?=\n\n|\n[A-Z])'
        match = re.search(pattern, markdown, re.IGNORECASE)
        
        if match:
            keywords_text = match.group(1).strip()
            # Split by comma or semicolon
            keywords = [k.strip() for k in re.split(r'[,;]', keywords_text)]
            # Filter out empty or very long "keywords"
            keywords = [k for k in keywords if 2 <= len(k) <= 50]
        
        return keywords[:10]  # Max 10 keywords
    
    def _extract_sections(self, markdown: str) -> List[PaperSection]:
        """
        Extract section structure.
        
        Detect:
        - Numbered sections (1. Introduction, 2.1 Method)
        - Named sections (## Methodology)
        - Section types (abstract, results, etc.)
        """
        sections = []
        
        # Find all Markdown headers
        # Pattern: ## Title or # Title or numbered sections
        pattern = r'^(#{1,4})\s+(?:(\d+(?:\.\d+)*)\s+)?(.+)$'
        
        for match in re.finditer(pattern, markdown, re.MULTILINE):
            hashes = match.group(1)
            number = match.group(2)
            title = match.group(3).strip()
            
            level = len(hashes)
            start_pos = match.start()
            
            # Classify section type
            section_type = self._classify_section(title)
            
            section = PaperSection(
                section_type=section_type,
                title=title,
                level=level,
                start_page=0,  # Will be calculated later if PDF available
                start_char=start_pos
            )
            
            sections.append(section)
        
        # Calculate end_char for each section
        for i in range(len(sections)):
            if i < len(sections) - 1:
                sections[i].end_char = sections[i + 1].start_char
            else:
                sections[i].end_char = len(markdown)
        
        # Detect equations, tables, figures in each section
        for section in sections:
            if section.start_char and section.end_char:
                section_text = markdown[section.start_char:section.end_char]
                section.has_equations = bool(re.search(r'\$\$|\$[^$]+\$', section_text))
                section.has_tables = bool(re.search(r'\|.+\|', section_text))
                section.has_figures = bool(re.search(r'\!\[.*?\]', section_text))
        
        logger.info(f"Detected {len(sections)} sections")
        return sections
    
    def _classify_section(self, title: str) -> str:
        """Classify section type based on title."""
        title_lower = title.lower()
        
        for section_type, pattern in self.SECTION_PATTERNS.items():
            if re.search(pattern, title_lower):
                return section_type
        
        return "other"
    
    def _extract_citations(self, markdown: str) -> List[Citation]:
        """
        Extract in-text citations.
        
        Patterns:
        1. Numbered: [1], [23], [1, 2, 3]
        2. Author-year: (Smith et al., 2020), (Jones, 2019)
        """
        citations = []
        
        # Pattern 1: Numbered citations [1]
        for match in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', markdown):
            citations.append(Citation(
                text=match.group(0),
                position=match.start(),
                citation_type="numbered"
            ))
        
        # Pattern 2: Author-year citations (Smith et al., 2020)
        pattern = r'\(([A-Z][a-z]+(?:\s+et al\.)?,?\s+\d{4}[a-z]?)\)'
        for match in re.finditer(pattern, markdown):
            citations.append(Citation(
                text=match.group(0),
                position=match.start(),
                citation_type="author-year"
            ))
        
        return citations
    
    def _count_references(self, markdown: str) -> int:
        """
        Count number of references in bibliography.
        
        Look for References section and count entries.
        """
        # Find References section
        pattern = r'#{1,3}\s*References\s*\n(.*?)(?=\n#{1,3}\s|\Z)'
        match = re.search(pattern, markdown, re.IGNORECASE | re.DOTALL)
        
        if match:
            references_text = match.group(1)
            # Count numbered entries: [1], [2], etc.
            numbered = len(re.findall(r'^\[\d+\]', references_text, re.MULTILINE))
            if numbered > 0:
                return numbered
            
            # Count lines starting with author names (fallback)
            lines = [l.strip() for l in references_text.split('\n') if l.strip()]
            return len([l for l in lines if re.match(r'^[A-Z]', l)])
        
        return 0
    
    def _extract_from_pdf(self, pdf_path: Path) -> PaperMetadata:
        """Extract additional metadata directly from PDF."""
        metadata = PaperMetadata()
        
        try:
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                metadata.total_pages = len(pdf.pages)
                
                # Extract PDF metadata
                if pdf.metadata:
                    if pdf.metadata.title:
                        metadata.title = pdf.metadata.title
                    if pdf.metadata.author:
                        # Split multiple authors
                        authors = pdf.metadata.author.split(',')
                        metadata.authors = [a.strip() for a in authors]
        
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
        
        return metadata
    
    def _merge_metadata(self, md_metadata: PaperMetadata, pdf_metadata: PaperMetadata) -> PaperMetadata:
        """Merge metadata from Markdown and PDF extraction."""
        # Prefer Markdown extraction (more reliable), but fill in gaps from PDF
        if not md_metadata.title and pdf_metadata.title:
            md_metadata.title = pdf_metadata.title
        if not md_metadata.authors and pdf_metadata.authors:
            md_metadata.authors = pdf_metadata.authors
        if not md_metadata.total_pages:
            md_metadata.total_pages = pdf_metadata.total_pages
        
        return md_metadata
    
    def _calculate_confidence(self, metadata: PaperMetadata) -> float:
        """
        Calculate confidence that this is a research paper.
        
        Scoring:
        - Has abstract: +0.3
        - Has references section: +0.2
        - Has authors: +0.15
        - Has structured sections: +0.15
        - Has year and venue: +0.1
        - Has citations: +0.1
        """
        score = 0.0
        
        if metadata.abstract:
            score += 0.3
        if metadata.num_references > 5:
            score += 0.2
        if len(metadata.authors) > 0:
            score += 0.15
        if len(metadata.sections) >= 3:
            score += 0.15
        if metadata.year and metadata.venue:
            score += 0.1
        if len(metadata.citations) > 10:
            score += 0.1
        
        return min(score, 1.0)
    
    def _determine_paper_type(self, metadata: PaperMetadata) -> str:
        """Determine paper type (conference, journal, arxiv, etc.)."""
        if metadata.arxiv_id:
            return "arxiv"
        if metadata.venue:
            venue_lower = metadata.venue.lower()
            if any(conf in venue_lower for conf in ["neurips", "icml", "cvpr", "acl", "aaai"]):
                return "conference"
            if any(jour in venue_lower for jour in ["ieee", "acm", "nature", "science"]):
                return "journal"
        
        # Default based on structure
        if len(metadata.sections) >= 5 and metadata.num_references > 20:
            return "journal"  # Longer papers usually journals
        
        return "unknown"


# Convenience function
def extract_paper_metadata(markdown: str, file_path: Optional[Path] = None) -> PaperMetadata:
    """
    Extract research paper metadata from Markdown.
    
    Usage:
        >>> metadata = extract_paper_metadata(markdown, Path("paper.pdf"))
        >>> if metadata.is_research_paper():
        >>>     print(f"Title: {metadata.title}")
        >>>     print(f"Authors: {', '.join(metadata.authors)}")
    """
    extractor = PaperMetadataExtractor()
    return extractor.extract_from_markdown(markdown, file_path)
