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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

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
        """
        Get section type for a given character position.

        The bounds are compared against None, not truth-tested: the first
        section of every document starts at character 0, and `if
        section.start_char` treated that as "no start recorded" and skipped it.
        """
        for section in self.sections:
            if section.start_char is None or section.end_char is None:
                continue
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
    
    # Section headers in research papers, in priority order: the first pattern
    # that matches wins, so the more specific phrasings come first.
    #
    # Order is load-bearing. "Experimental Results" is a results section, so
    # `results` is tested before `experiments`; "Conclusion and Future Work" is
    # a conclusion, so `conclusion` is tested before both. Testing
    # `abstract: (abstract|summary)` first, as this used to, classified
    # "Summary and Future Work" as an abstract.
    #
    # The vocabulary is wider than the original nine patterns because real
    # papers do not name their sections after the canonical IMRaD labels. On
    # the QASPER corpus the old set left 46.9% of chunks tagged "other" --
    # "Data set", "Setup", "Ablation Study", "Error Analysis", "Baselines" all
    # fell through -- and section routing cannot work over labels that mostly
    # say nothing.
    SECTION_PATTERNS = {
        "references": r"\b(references|bibliograph(y|ies)|works cited)\b",
        "acknowledgements": r"\backnowledge?ments?\b",
        "appendix": r"\b(appendix|appendices|supplementary)\b",
        "abstract": r"\babstract\b",
        "related_work": (
            r"\b(related work|literature review|prior work|previous work|"
            r"related literature|state of the art)\b"
        ),
        "conclusion": (
            r"\b(conclusions?|concluding remarks|future work|future directions|"
            r"limitations?|takeaways?|summary)\b"
        ),
        "results": (
            r"\b(results?|findings|error analysis|main finding)\b"
        ),
        "experiments": (
            r"\b(experiments?|experimental|evaluation|validation|"
            r"(experimental |training |implementation )?(setup|settings?|details)|"
            r"ablations?|baselines?|hyper-?parameters?|"
            r"(evaluation |training )?(metrics?|protocol|procedure)|"
            r"case stud(y|ies)|user study|human evaluation)\b"
        ),
        "dataset": (
            r"\b(datasets?|data set|corpus|corpora|data collection|"
            r"data description|data preparation|annotation(s| scheme| process)?|"
            r"data)\b"
        ),
        "methodology": (
            r"\b(method(s|ology|ologies)?|approach(es)?|model(s|ling|ing)?|"
            r"architectures?|systems?|algorithms?|frameworks?|networks?|"
            r"formulation|preliminaries|notation|"
            r"(task|problem) (definition|statement|formulation)|"
            r"our (model|method|approach|system))\b"
        ),
        "discussion": r"\b(discussion|analysis|interpretation|qualitative)\b",
        "introduction": r"\b(introduction|motivation|overview|background)\b",
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
        # Authors are bounded below the title, and the year uses the arXiv id
        # as a fallback signal, so both depend on what is extracted above them.
        metadata.authors = self._extract_authors(markdown, metadata.title)
        metadata.doi = self._extract_doi(markdown)
        metadata.arxiv_id = self._extract_arxiv_id(markdown)
        metadata.year = self._extract_year(markdown, metadata.arxiv_id)
        metadata.venue = self._extract_venue(markdown)
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
    
    # Journal furniture that appears above the real title on the first page.
    _BOILERPLATE = re.compile(
        r'^\s*('
        r'RESEARCH(\s+ARTICLE)?|REVIEW|ARTICLE|ORIGINAL\s+(RESEARCH|ARTICLE|PAPER)|'
        r'SHORT\s+(REPORT|COMMUNICATION)|METHODOLOGY|SURVEY|EDITORIAL|PREPRINT|'
        r'OPEN\s+ACCESS|CORRESPONDENCE|ABSTRACT|KEYWORDS|INTRODUCTION|'
        r'CHECK\s+FOR\s+UPDATES|SUPPLEMENTARY|©.*'
        r')\s*$',
        re.IGNORECASE,
    )

    # Running head: "Saadi et al. Journal of Big Data (2025) 12:84"
    _RUNNING_HEAD = re.compile(
        r'et\s+al\.|'
        r'\(\d{4}\)\s*\d+\s*:\s*\d+|'
        r'\bvol\.?\s*\d+|\bno\.?\s*\d+|\bpp\.?\s*\d+|'
        r'^\s*page\s+\d+|^\s*\d+\s*$',
        re.IGNORECASE,
    )

    # Author line: names carrying affiliation markers, e.g. "Aicha Saadi1*, Noureddine Abghour1"
    _AUTHOR_LINE = re.compile(r'[A-Za-z]{2,}\s*\d{1,2}\s*[*†‡§]?\s*(,|and\b|$)')

    def _is_title_candidate(self, line: str) -> bool:
        """Whether a line could form part of a paper title."""
        if not (4 <= len(line) <= 300):
            return False
        low = line.lower()
        if 'http' in low or 'www.' in low or 'doi.org' in low or low.startswith('doi'):
            return False
        if '@' in line:                      # correspondence email
            return False
        if self._BOILERPLATE.match(line):
            return False
        if self._RUNNING_HEAD.search(line):
            return False
        if self._AUTHOR_LINE.search(line):   # we've reached the author block
            return False
        if not any(c.isalpha() for c in line):
            return False
        return True

    def _extract_title(self, markdown: str) -> Optional[str]:
        """
        Extract paper title.

        Strategy:
        1. First Markdown `# ` header (most reliable when present)
        2. Otherwise scan the top of the document, skipping journal furniture
           (running heads, DOIs, "RESEARCH"/"Open Access" banners), and join
           the consecutive lines of the title block — titles in converted PDFs
           are frequently wrapped across two or three lines.
        """
        top_section = markdown[:2000]

        # Strategy 1: First # header (most reliable for Markdown)
        match = re.search(r'^#\s+(.+)$', top_section, re.MULTILINE)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'[*_]+$', '', title).strip()
            if 10 < len(title) < 300:
                return title

        # Strategy 2: Accumulate the title block from the top of the page
        block: List[str] = []
        for raw in top_section.split('\n')[:40]:
            line = raw.strip()

            if not line:
                # A blank line ends the title once we have enough of one.
                if len(' '.join(block)) >= 20:
                    break
                block = []          # false start; keep looking
                continue

            if self._is_title_candidate(line):
                block.append(line)
            elif block:
                break               # hit authors or boilerplate after the title
            # else: still in the header furniture, keep skipping

        title = ' '.join(block).strip()
        title = re.sub(r'\s+', ' ', title)
        title = re.sub(r'[*_]+$', '', title).strip()

        if 10 < len(title) < 300:
            return title
        return None
    
    # Words that mark a capitalized phrase as an organization, journal, or
    # section heading rather than a person's name.
    _NON_NAME_WORDS = {
        'journal', 'data', 'access', 'open', 'research', 'review', 'article',
        'abstract', 'keywords', 'introduction', 'conclusion', 'references',
        'university', 'faculty', 'department', 'institute', 'college', 'school',
        'laboratory', 'labs', 'lab', 'sciences', 'science', 'center', 'centre',
        'hospital', 'academy', 'society', 'association', 'foundation',
        'springer', 'elsevier', 'wiley', 'nature', 'press', 'publishing',
        'license', 'licence', 'creative', 'commons', 'copyright',
        'correspondence', 'author', 'authors', 'received', 'accepted',
        'published', 'available', 'online', 'supplementary', 'figure', 'table',
        'big', 'international', 'conference', 'proceedings', 'transactions',
        'learning', 'network', 'networks', 'systems', 'control', 'traffic',
    }

    def _is_not_a_person(self, name: str) -> bool:
        """Whether a capitalized phrase is an organization/heading, not a person."""
        words = {w.lower().strip('.,') for w in name.split()}
        return bool(words & self._NON_NAME_WORDS)

    # ------------------------------------------------------------------
    # Authors
    # ------------------------------------------------------------------

    # One token of a personal name: an initial ("J", "J."), a nobiliary
    # particle ("van", "de", "al-"), or a capitalized word, including
    # hyphenated and apostrophised surnames and accented letters.
    _NAME_TOKEN = re.compile(
        r"^(?:"
        r"[A-ZÀ-Þ]\.?"
        r"|(?:[Dd]e|[Dd]a|[Dd]i|[Dd]el|[Dd]er|[Dd]en|[Dd]os|[Vv]an|[Vv]on|[Ll]a|[Ll]e"
        r"|[Bb]in|[Ii]bn|[Aa]l|[Ss]an|[Ss]t\.?)"
        r"|(?:Mc|Mac|O['’])?"
        r"[A-ZÀ-Þ][a-zß-ÿ]+"
        r"(?:[-'’][A-ZÀ-Þa-zß-ÿ][a-zß-ÿ]*)*"
        r")$"
    )

    # Particles and initials cannot stand in for a surname on their own.
    _NAME_PARTICLES = {
        'de', 'da', 'di', 'del', 'der', 'den', 'dos', 'van', 'von', 'la', 'le',
        'bin', 'ibn', 'al', 'san', 'st', 'st.',
    }

    # Footnote and affiliation markers hung off author names.
    _AUTHOR_MARKERS = re.compile(
        r'[*†‡§¶∗¹²³⁰-⁹]|\d'
    )

    # The line that ends the front matter: the first real section.
    _FRONT_MATTER_END = re.compile(
        r'^\s*(?:#{1,6}\s*)?(?:\d+\.?\s*)?('
        r'abstract|summary|keywords?|index\s+terms|introduction|'
        r'correspondence|received:|accepted:|published:|citation:'
        r')\b',
        re.IGNORECASE,
    )

    _EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')

    @staticmethod
    def _normalize_for_compare(text: str) -> str:
        """Lowercase alphanumeric skeleton, for comparing a line to the title."""
        return re.sub(r'[^a-z0-9]+', '', text.lower())

    def _author_region(self, markdown: str, title: Optional[str]) -> List[str]:
        """
        The lines that could hold the author block.

        Authors sit between the title and the first front-matter section, so
        the search is bounded on both sides. Bounding it at the top is what
        stops the title itself being read as a name — the failure that gave a
        paper with no author line authors like
        ["New Multimodal Benchmark Dataset"].

        Args:
            markdown: Converted document text
            title: Extracted title, walked past before collecting

        Returns:
            Candidate author-block lines, in document order
        """
        lines = markdown[:2500].split('\n')
        title_skeleton = self._normalize_for_compare(title) if title else ''

        region: List[str] = []
        consumed_title = not bool(title_skeleton)
        seen_skeleton = ''

        for raw in lines[:60]:
            line = raw.strip()
            if not line:
                continue
            if self._FRONT_MATTER_END.match(line):
                break

            stripped = re.sub(r'^#{1,6}\s*', '', line).strip()

            # Walk past the title, which converters often wrap over several
            # lines, before collecting anything.
            if not consumed_title:
                seen_skeleton += self._normalize_for_compare(stripped)
                if title_skeleton and title_skeleton in seen_skeleton:
                    consumed_title = True
                continue

            # A markdown header after the title starts the body.
            if line.startswith('#'):
                break

            region.append(line)
            if len(region) >= 8:
                break

        return region

    def _is_person_name(self, name: str) -> bool:
        """Whether a string has the shape of a personal name."""
        if not (3 <= len(name) <= 60):
            return False

        tokens = name.split()
        if not (2 <= len(tokens) <= 5):
            return False
        if any(not self._NAME_TOKEN.match(token) for token in tokens):
            return False

        # At least one full word that is neither an initial nor a particle --
        # a surname. "P. F. Brown" qualifies; "J. R." and "van der" do not.
        substantive = [
            token for token in tokens
            if len(token.rstrip('.')) > 1
            and token.lower().rstrip('.') not in self._NAME_PARTICLES
        ]
        if not substantive:
            return False

        return not self._is_not_a_person(name)

    def _parse_author_line(self, line: str) -> Optional[List[str]]:
        """
        Parse one line as a list of author names.

        Returns None when the line is not an author line at all, so the caller
        can tell "the block has not started" from "the block has ended".
        """
        # Strip markdown emphasis, list markers, emails and bracketed asides.
        candidate = re.sub(r'[*_`]{1,3}', '', line).strip()
        candidate = re.sub(r'^[-•]\s*', '', candidate)
        candidate = self._EMAIL.sub(' ', candidate)
        candidate = re.sub(r'\(.*?\)|\[.*?\]', ' ', candidate)
        if not candidate.strip():
            return None

        names: List[str] = []
        for part in re.split(r'\s*(?:,|;|\band\b|&)\s*', candidate):
            # Drop affiliation markers: superscripts, footnote symbols, digits.
            name = self._AUTHOR_MARKERS.sub('', part).strip(' .·')
            name = re.sub(r'\s+', ' ', name)
            if not name:
                continue
            if not self._is_person_name(name):
                return None
            names.append(name)

        return names or None

    def _extract_authors(
        self,
        markdown: str,
        title: Optional[str] = None,
    ) -> List[str]:
        """
        Extract author names from the block between the title and the abstract.

        A capitalized phrase is accepted as an author list only when the block
        looks like one: two or more names, or names carrying affiliation
        markers, or an email in the block. A document whose converted text has
        no author line gets an empty list, which is the truthful answer.

        The previous version scanned the first 2000 characters with a bare
        Title-Case regex, so it returned fragments of the title as authors.
        Those fragments then flowed into every citation the document produced,
        and are why citation accuracy could not be scored on QASPER.

        Args:
            markdown: Converted document text
            title: Extracted title, used to bound the search below it

        Returns:
            Author names in document order, empty when none can be identified
        """
        region = self._author_region(markdown, title)
        if not region:
            return []

        block_has_email = any(self._EMAIL.search(line) for line in region)

        authors: List[str] = []
        marker_seen = False

        for line in region:
            names = self._parse_author_line(line)
            if names is None:
                if authors:
                    break        # the author block has ended
                continue         # not started yet; keep looking
            if self._AUTHOR_MARKERS.search(line):
                marker_seen = True
            authors.extend(names)
            if len(authors) >= 40:
                break

        if not authors:
            return []

        # Require positive evidence that this really is an author list. A
        # single bare name with no marker and no email is more often a stray
        # line of the title than an author, and a wrong author is worse than
        # no author: citation validation checks generated citations against
        # these surnames.
        if len(authors) < 2 and not (marker_seen or block_has_email):
            logger.debug("Author candidate rejected: no author-block signal")
            return []

        seen = set()
        unique_authors = []
        for author in authors:
            if author not in seen:
                seen.add(author)
                unique_authors.append(author)

        return unique_authors[:20]  # Max 20 authors (reasonable limit)

    # ------------------------------------------------------------------
    # Year
    # ------------------------------------------------------------------

    # A year attached to something that actually dates a publication.
    _YEAR_CUE = re.compile(
        r'(?:©|\(c\)|copyright|published(?:\s+online)?|received|accepted|'
        r'revised|in\s+press|to\s+appear|proceedings\s+of|preprint|'
        r'vol\.?\s*\d+|volume\s*\d+)'
        r'[^\n]{0,80}?\b((?:19|20)\d{2})\b',
        re.IGNORECASE,
    )

    # Journal running head: "Journal of Big Data (2025) 12:84"
    _YEAR_RUNNING_HEAD = re.compile(r'\((\d{4})\)\s*\d+\s*[:(]')

    # A dateline that stands at the start of its own line. Some layouts put
    # "Received: ... Accepted: ... Published: ..." below the abstract, so this
    # narrower set is allowed to look past the front matter, where the broader
    # cue set would match prose ("documents published in 2003").
    _YEAR_DATELINE = re.compile(
        r'^\s*(?:©|\(c\)|copyright|published|received|accepted|revised)\b'
        r'[^\n]{0,60}?\b((?:19|20)\d{2})\b',
        re.IGNORECASE | re.MULTILINE,
    )

    def _extract_year(
        self,
        markdown: str,
        arxiv_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Extract the publication year.

        Signals are tried in order of what each is worth: an explicit
        publication cue ("Published 12 March 2024", "© 2019"), a journal
        running head, the arXiv identifier, then the most frequent plausible
        year in the front matter.

        Returns None when nothing supports a year. Callers must not turn that
        into 0, which reads as a real value and renders as "(Smith, 0)".

        Args:
            markdown: Converted document text
            arxiv_id: Extracted arXiv identifier, which encodes the year

        Returns:
            Four-digit year, or None when undetermined
        """
        top_section = markdown[:2500]
        current_year = datetime.now().year

        def plausible(value: int) -> bool:
            return 1970 <= value <= current_year + 1

        # Everything above the first real section. Publication dates live here;
        # the abstract below it discusses years that belong to the work, not to
        # the paper.
        front_matter = top_section
        for line_match in re.finditer(r'^.*$', top_section, re.MULTILINE):
            if self._FRONT_MATTER_END.match(line_match.group(0).strip()):
                front_matter = top_section[:line_match.start()]
                break

        for pattern, scope in (
            (self._YEAR_CUE, front_matter),
            (self._YEAR_RUNNING_HEAD, front_matter),
            (self._YEAR_DATELINE, top_section),
        ):
            match = pattern.search(scope)
            if match and plausible(int(match.group(1))):
                return int(match.group(1))

        if arxiv_id:
            # 2103.12345 -> 2021; 0706.0001 -> 2007.
            prefix = arxiv_id.split('.')[0]
            if len(prefix) == 4 and prefix.isdigit():
                year = 2000 + int(prefix[:2])
                if plausible(year):
                    return year

        # Fall back to the most frequent plausible year, breaking ties toward
        # the most recent -- still over the front matter only. Scanning the
        # whole top section, as this used to, reads a year out of the
        # abstract's prose ("trained on 2015 data") and reports it as the
        # publication year. Documents with no front matter get None.
        years = [
            int(y) for y in re.findall(r'\b((?:19|20)\d{2})\b', front_matter)
        ]
        years = [y for y in years if plausible(y)]
        if years:
            counts = Counter(years)
            most_common = max(counts.values())
            return max(y for y, n in counts.items() if n == most_common)

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

        # PDF -> Markdown conversion usually yields no '#' headers at all, so
        # fall back to detecting headings that stand alone on their own line.
        if not sections:
            sections = self._extract_plaintext_sections(markdown)

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
    
    def _extract_plaintext_sections(self, markdown: str) -> List[PaperSection]:
        """
        Detect section headings in text that has no Markdown headers.

        Converters flatten PDFs into plain text, so headings survive only as
        short standalone lines ("Introduction", "3.2 Reward design"). A line is
        treated as a heading when it sits on its own, is short, carries no
        sentence punctuation, and either names a known section or is numbered.

        Returns:
            Sections ordered by position in the document
        """
        sections: List[PaperSection] = []
        seen_types: set[str] = set()

        # "3", "3.2", "IV" prefixes are strong heading signals in papers.
        numbered = re.compile(r'^\s*(\d+(?:\.\d+)*|[IVX]{1,5})[.)]?\s+(.{2,70})$')

        offset = 0
        lines = markdown.split('\n')
        for i, raw in enumerate(lines):
            line_start = offset
            offset += len(raw) + 1  # +1 for the newline consumed by split

            line = raw.strip()
            if not (2 <= len(line) <= 80):
                continue
            # Headings stand alone: require blank space above (or document top).
            if i > 0 and lines[i - 1].strip():
                continue
            # Sentences and list items are not headings.
            if line.endswith(('.', ',', ';', ':')) or line.startswith(('-', '*', '|')):
                continue
            if len(line.split()) > 10:
                continue

            # Figure/table captions read like headings but are not sections.
            if re.match(r'^(fig(ure)?\.?|table|algorithm|eq(uation)?\.?|scheme)\b',
                        line, re.IGNORECASE):
                continue

            m = numbered.match(line)
            title = m.group(2).strip() if m else line
            level = 2 if (m and '.' in (m.group(1) or '')) else 1

            # Affiliation lines are numbered like sections ("1 LIS Labs, ...").
            if self._is_not_a_person(title) and self._classify_section(title) == "other":
                continue

            section_type = self._classify_section(title)

            # Without a number to vouch for it, only accept known section names.
            if section_type == "other" and not m:
                continue
            # Unnumbered headings repeat in running heads; keep the first only.
            if not m and section_type in seen_types:
                continue
            seen_types.add(section_type)

            sections.append(PaperSection(
                section_type=section_type,
                title=title,
                level=level,
                start_page=0,
                start_char=line_start,
            ))

        return sections

    def _classify_section(self, title: str) -> str:
        """Classify section type based on title."""
        title_lower = title.lower()

        for section_type, pattern in self.SECTION_PATTERNS.items():
            if re.search(pattern, title_lower):
                return section_type

        return "other"

    # Separator used in the chunker's header_path, e.g.
    # "Experiments > Baseline Models".
    HEADER_PATH_SEPARATOR = " > "

    def classify_header_path(
        self,
        header_path: str,
        document_title: Optional[str] = None,
    ) -> str:
        """
        Classify a chunk's section from its chain of enclosing headers.

        The outermost heading that can be classified wins. A paper's top-level
        heading declares what a region is for and its subsections refine it, so
        "Experiments > Baseline Models" belongs to the experiments, and
        "Model > Background" belongs to the method rather than the
        introduction, which is what classifying the leaf alone would say.

        The document title is dropped first when it heads the path. It is a
        heading like any other to the chunker, but classifying it applies one
        label to the entire paper: a paper called "A Neural Model for Question
        Answering" would have every chunk tagged `methodology`, because the
        title contains "model".

        Args:
            header_path: Headers from outermost to innermost, separated by
                HEADER_PATH_SEPARATOR. A bare heading is accepted too.
            document_title: Title to ignore when it leads the path

        Returns:
            A section type, or "other" when no heading in the chain matches
        """
        if not header_path:
            return "other"

        parts = [
            part.strip()
            for part in header_path.split(self.HEADER_PATH_SEPARATOR)
            if part.strip()
        ]

        if (
            document_title
            and len(parts) > 1
            and self._normalize_for_compare(parts[0])
            == self._normalize_for_compare(document_title)
        ):
            parts = parts[1:]

        for part in parts:
            section_type = self._classify_section(part)
            if section_type != "other":
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
            lines = [ln.strip() for ln in references_text.split('\n') if ln.strip()]
            return len([ln for ln in lines if re.match(r'^[A-Z]', ln)])
        
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
        - Has abstract: +0.30
        - Has structured sections: +0.20
        - Has references section: +0.15
        - Has a title: +0.10
        - Has authors: +0.10
        - Has citations: +0.10
        - Has year and venue: +0.05

        Authors carry little weight on purpose. Author extraction now returns
        an empty list rather than guessing from the title, and plenty of real
        papers reach the system as converted text with no author line at all --
        every reconstructed QASPER paper does. Weighting authors at 0.15, as
        this did, put those papers below the 0.5 threshold, which switches off
        chunk enrichment and leaves every chunk with no section_type. The
        structural signals (abstract, sections, references) are what actually
        distinguish a paper from a memo.
        """
        score = 0.0

        if metadata.abstract:
            score += 0.30
        if len(metadata.sections) >= 3:
            score += 0.20
        if metadata.num_references > 5:
            score += 0.15
        if metadata.title:
            score += 0.10
        if len(metadata.authors) > 0:
            score += 0.10
        if len(metadata.citations) > 10:
            score += 0.10
        if metadata.year and metadata.venue:
            score += 0.05

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
