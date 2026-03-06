"""
Tests for Citation Formatter.

Tests Phase 4.2: Citation formatting in multiple styles.
"""

import pytest
from docusense.generation.citation_formatter import (
    CitationFormatter,
    CitationStyle,
    FormattedCitation,
    format_citation
)


# ==============================================================================
# Test Data
# ==============================================================================

def make_bert_source():
    """Create BERT paper source metadata."""
    return {
        "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "year": 2018,
        "venue": "NAACL",
        "section_type": "results",
        "doi": "10.18653/v1/N19-1423",
        "chunk_id": "chunk_001",
        "document_id": "doc_bert",
        "score": 0.92
    }


def make_gpt2_source():
    """Create GPT-2 paper source metadata."""
    return {
        "paper_title": "Language Models are Unsupervised Multitask Learners",
        "authors": ["Alec Radford", "Jeffrey Wu", "Rewon Child"],
        "year": 2019,
        "venue": "OpenAI",
        "section_type": "abstract",
        "doi": "",
        "chunk_id": "chunk_003",
        "document_id": "doc_gpt2",
        "score": 0.78
    }


def make_multi_sources():
    """Create multiple sources including duplicates from same paper."""
    return [
        make_bert_source(),
        {
            **make_bert_source(),
            "section_type": "methodology",  # Different section, same paper
            "chunk_id": "chunk_002",
            "score": 0.85
        },
        make_gpt2_source(),
    ]


# ==============================================================================
# Inline Citation Tests
# ==============================================================================

class TestInlineCitation:
    """Tests for inline citation formatting."""
    
    def test_single_author(self):
        """Test citation with single author."""
        formatter = CitationFormatter()
        meta = {"authors": ["John Smith"], "year": 2020}
        result = formatter.format_inline_citation(meta)
        
        assert "Smith" in result
        assert "2020" in result
        assert result.startswith("(")
        assert result.endswith(")")
    
    def test_two_authors(self):
        """Test citation with two authors."""
        formatter = CitationFormatter()
        meta = {"authors": ["John Smith", "Jane Doe"], "year": 2020}
        result = formatter.format_inline_citation(meta)
        
        assert "Smith" in result
        assert "Doe" in result
        assert "2020" in result
    
    def test_three_plus_authors_et_al(self):
        """Test et al. for 3+ authors."""
        formatter = CitationFormatter()
        meta = {
            "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
            "year": 2018
        }
        result = formatter.format_inline_citation(meta)
        
        assert "Devlin" in result
        assert "et al." in result
        assert "2018" in result
    
    def test_with_section(self):
        """Test citation with section type."""
        formatter = CitationFormatter()
        meta = {
            "authors": ["Jacob Devlin"],
            "year": 2018,
            "section_type": "results"
        }
        result = formatter.format_inline_citation(meta)
        
        assert "Results" in result
    
    def test_with_page_number(self):
        """Test citation with page number."""
        formatter = CitationFormatter()
        meta = {
            "authors": ["Jacob Devlin"],
            "year": 2018,
            "page": "9"
        }
        result = formatter.format_inline_citation(meta)
        
        assert "p.9" in result
    
    def test_no_authors(self):
        """Test citation without authors."""
        formatter = CitationFormatter()
        meta = {"year": 2020}
        result = formatter.format_inline_citation(meta)
        
        assert "2020" in result
    
    def test_no_year(self):
        """Test citation without year."""
        formatter = CitationFormatter()
        meta = {"authors": ["John Smith"]}
        result = formatter.format_inline_citation(meta)
        
        assert "n.d." in result


# ==============================================================================
# Reference List Tests
# ==============================================================================

class TestReferenceList:
    """Tests for reference list formatting."""
    
    def test_format_reference_list(self):
        """Test full reference list generation."""
        formatter = CitationFormatter()
        sources = make_multi_sources()
        
        ref_list = formatter.format_reference_list(sources)
        
        assert "References:" in ref_list
        assert "[1]" in ref_list
        assert "[2]" in ref_list
        assert "BERT" in ref_list
        assert "Language Models" in ref_list
    
    def test_deduplication(self):
        """Test that duplicate papers are merged."""
        formatter = CitationFormatter()
        sources = make_multi_sources()
        
        citations = formatter.format_sources(sources)
        
        # 3 sources but only 2 unique papers
        assert len(citations) == 2
    
    def test_reference_format(self):
        """Test individual reference formatting."""
        formatter = CitationFormatter()
        sources = [make_bert_source()]
        
        citations = formatter.format_sources(sources)
        ref = citations[0].reference
        
        assert "[1]" in ref
        assert "Devlin" in ref
        assert "(2018)" in ref
        assert "BERT" in ref
        assert "NAACL" in ref
    
    def test_empty_sources(self):
        """Test with empty source list."""
        formatter = CitationFormatter()
        ref_list = formatter.format_reference_list([])
        
        assert ref_list == ""


# ==============================================================================
# BibTeX Tests
# ==============================================================================

class TestBibTeX:
    """Tests for BibTeX export."""
    
    def test_bibtex_entry(self):
        """Test BibTeX entry generation."""
        formatter = CitationFormatter()
        sources = [make_bert_source()]
        
        citations = formatter.format_sources(sources)
        bibtex = citations[0].bibtex
        
        assert "@inproceedings{" in bibtex or "@article{" in bibtex
        assert "title={BERT" in bibtex
        assert "author={Jacob Devlin" in bibtex
        assert "year={2018}" in bibtex
    
    def test_bibtex_export(self):
        """Test full BibTeX export."""
        formatter = CitationFormatter()
        sources = make_multi_sources()
        
        bibtex = formatter.format_bibtex_export(sources)
        
        # Should have 2 entries (deduplicated)
        assert bibtex.count("@") == 2
        assert "BERT" in bibtex
        assert "Language Models" in bibtex
    
    def test_bibtex_key_generation(self):
        """Test BibTeX key is properly formatted."""
        key = CitationFormatter._generate_bibtex_key(
            authors=["Jacob Devlin"],
            year=2018,
            title="BERT: Pre-training of Deep Bidirectional Transformers"
        )
        
        assert "devlin" in key
        assert "2018" in key
        assert "bert" in key.lower()
    
    def test_bibtex_empty_sources(self):
        """Test BibTeX export with empty sources."""
        formatter = CitationFormatter()
        bibtex = formatter.format_bibtex_export([])
        
        assert bibtex == ""


# ==============================================================================
# FormattedCitation Tests
# ==============================================================================

class TestFormattedCitation:
    """Tests for FormattedCitation dataclass."""
    
    def test_citation_fields(self):
        """Test all fields of FormattedCitation are populated."""
        formatter = CitationFormatter()
        sources = [make_bert_source()]
        
        citations = formatter.format_sources(sources)
        c = citations[0]
        
        assert c.citation_num == 1
        assert c.inline.startswith("(")
        assert c.inline.endswith(")")
        assert "[1]" in c.reference
        assert "@" in c.bibtex
        assert c.paper_title == "BERT: Pre-training of Deep Bidirectional Transformers"
        assert len(c.authors) == 4
        assert c.year == 2018
        assert c.venue == "NAACL"
    
    def test_section_tracking(self):
        """Test that sections from multiple chunks are tracked."""
        formatter = CitationFormatter()
        sources = make_multi_sources()
        
        citations = formatter.format_sources(sources)
        
        # BERT paper should have both results and methodology sections
        bert_citation = [c for c in citations if "BERT" in c.paper_title][0]
        assert bert_citation.section_type is not None
        assert "results" in bert_citation.section_type or "methodology" in bert_citation.section_type


# ==============================================================================
# Convenience Function Tests
# ==============================================================================

class TestConvenienceFunction:
    """Tests for the format_citation convenience function."""
    
    def test_format_citation(self):
        """Test quick citation formatting."""
        result = format_citation(make_bert_source())
        
        assert "Devlin" in result
        assert "2018" in result
        assert result.startswith("(")
