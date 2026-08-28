"""
Tests for research-paper metadata extraction.

These cover the failures that made the extracted metadata untrustworthy:
title text returned as author names, an undetermined year reported as a real
one, and section labels so sparse that section routing had nothing to route on.
"""

from __future__ import annotations

import pytest

from docusense.ingestion.paper_metadata import PaperMetadataExtractor


@pytest.fixture(scope="module")
def extractor() -> PaperMetadataExtractor:
    return PaperMetadataExtractor()


# ==============================================================================
# Authors
# ==============================================================================

SPRINGER_FRONT_MATTER = """Saadi et al. Journal of Big Data (2025) 12:84
https://doi.org/10.1186/s40537-025-01104-x

RESEARCH

Open Access

A New Multimodal Benchmark Dataset for Fake
News Detection in Low-Resource Languages

Aicha Saadi1*, Noureddine Abghour1 and Zouhair Chiba1

*Correspondence: a.saadi@univ.ma

1 LIS Labs, Faculty of Sciences, Hassan II University, Casablanca, Morocco

Abstract
We present a new multimodal benchmark dataset for fake news detection.
"""

# A reconstructed QASPER paper: real structure, no author line anywhere.
NO_AUTHOR_LINE = """# Political Speech Generation

## Abstract

In this report we present a system that can generate political speeches.

## Introduction

Many political speeches show the same structures.
"""


class TestAuthorExtraction:
    def test_reads_a_real_author_line(self, extractor):
        meta = extractor.extract_from_markdown(SPRINGER_FRONT_MATTER)
        assert meta.authors == [
            "Aicha Saadi",
            "Noureddine Abghour",
            "Zouhair Chiba",
        ]

    def test_a_document_with_no_author_line_has_no_authors(self, extractor):
        """
        The regression this file exists for. A bare Title-Case regex over the
        first 2000 characters returned fragments of the title as authors --
        ["New Multimodal Benchmark Dataset"] -- and those fragments then flowed
        into every citation the document produced.
        """
        meta = extractor.extract_from_markdown(NO_AUTHOR_LINE)
        assert meta.authors == []

    def test_the_title_is_never_returned_as_an_author(self, extractor):
        markdown = (
            "A New Multimodal Benchmark Dataset\n"
            "for Fake News Detection\n"
            "\n"
            "Abstract\n"
            "This paper introduces a dataset.\n"
        )
        meta = extractor.extract_from_markdown(markdown)
        assert meta.authors == []

    def test_handles_initials_and_compound_surnames(self, extractor):
        markdown = (
            "# Statistical Machine Translation\n\n"
            "P. F. Brown, S. A. Della Pietra, V. J. Della Pietra\n\n"
            "## Abstract\nAn overview.\n"
        )
        meta = extractor.extract_from_markdown(markdown)
        assert meta.authors == [
            "P. F. Brown",
            "S. A. Della Pietra",
            "V. J. Della Pietra",
        ]

    def test_accepts_a_lone_author_carrying_a_footnote_marker(self, extractor):
        markdown = (
            "# Deep Learning for Protein Folding\n\n"
            "John Jumper*\n\n"
            "*DeepMind, London, UK\n\n"
            "## Abstract\nWe describe AlphaFold.\n"
        )
        meta = extractor.extract_from_markdown(markdown)
        assert meta.authors == ["John Jumper"]

    def test_affiliations_are_not_mistaken_for_authors(self, extractor):
        meta = extractor.extract_from_markdown(SPRINGER_FRONT_MATTER)
        joined = " ".join(meta.authors).lower()
        assert "university" not in joined
        assert "faculty" not in joined


# ==============================================================================
# Year
# ==============================================================================

class TestYearExtraction:
    def test_prefers_an_explicit_publication_cue(self, extractor):
        markdown = (
            "# Efficient Transformers\n\n"
            "Jane Doe, Richard Roe\n\n"
            "Copyright (c) 2019 Association for Computational Linguistics.\n\n"
            "## Abstract\nWe build on work from 2015 and 2017.\n"
        )
        assert extractor.extract_from_markdown(markdown).year == 2019

    def test_derives_the_year_from_an_arxiv_identifier(self, extractor):
        markdown = (
            "# Attention Is All You Need\n\n"
            "Ashish Vaswani, Noam Shazeer\n\n"
            "arXiv:1706.03762v5\n\n"
            "## Abstract\nThe dominant sequence transduction models.\n"
        )
        assert extractor.extract_from_markdown(markdown).year == 2017

    def test_reads_a_journal_running_head(self, extractor):
        assert extractor.extract_from_markdown(SPRINGER_FRONT_MATTER).year == 2025

    def test_an_undetermined_year_is_none_not_zero(self, extractor):
        """
        None means "not determined". Zero is a value, and it renders in a
        citation as "(Smith, 0)" rather than "(Smith, n.d.)".
        """
        meta = extractor.extract_from_markdown(NO_AUTHOR_LINE)
        assert meta.year is None

    def test_does_not_read_a_year_out_of_the_abstract(self, extractor):
        """A year discussed in the abstract is not the publication year."""
        markdown = (
            "# A Corpus of Historical Text\n\n"
            "## Abstract\n"
            "We collect documents published in 2003, 2003 and 2003.\n"
        )
        assert extractor.extract_from_markdown(markdown).year is None


# ==============================================================================
# Section classification
# ==============================================================================

class TestSectionClassification:
    @pytest.mark.parametrize(
        "heading,expected",
        [
            ("Introduction", "introduction"),
            ("Related Work", "related_work"),
            ("Data set", "dataset"),
            ("Datasets", "dataset"),
            ("Experimental Setup", "experiments"),
            ("Ablation Study", "experiments"),
            ("Baselines", "experiments"),
            ("Human Evaluation", "experiments"),
            ("Results", "results"),
            ("Experimental Results", "results"),
            ("Error Analysis", "results"),
            ("Alternative Methods", "methodology"),
            ("Our Model", "methodology"),
            ("Problem Formulation", "methodology"),
            ("Discussion", "discussion"),
            ("Conclusion and Future Work", "conclusion"),
            ("Limitations", "conclusion"),
            ("Acknowledgements", "acknowledgements"),
            ("References", "references"),
            ("Appendix A", "appendix"),
        ],
    )
    def test_classifies_headings_seen_in_real_papers(
        self, extractor, heading, expected
    ):
        assert extractor._classify_section(heading) == expected

    def test_conclusion_beats_abstract_on_a_summary_heading(self, extractor):
        """
        "abstract: (abstract|summary)" was tested first, so
        "Summary and Future Work" came back as an abstract.
        """
        assert extractor._classify_section("Summary and Future Work") == "conclusion"

    def test_a_subsection_inherits_from_its_parent(self, extractor):
        """
        "Baseline Models" says nothing on its own. Its parent does.
        """
        assert (
            extractor.classify_header_path("Experiments > Baseline Models")
            == "experiments"
        )

    def test_the_outer_section_wins_over_a_misleading_leaf(self, extractor):
        """
        "Background" alone reads as an introduction; under "Model" it is part
        of the method.
        """
        assert extractor.classify_header_path("Model > Background") == "methodology"

    def test_the_document_title_is_not_classified(self, extractor):
        """
        The title heads every header path. Classifying it labels the whole
        paper: this title contains "model", so every chunk would be tagged
        methodology.
        """
        title = "A Neural Model for Question Answering"
        assert (
            extractor.classify_header_path(
                f"{title} > Related Work", document_title=title
            )
            == "related_work"
        )
        assert (
            extractor.classify_header_path(
                f"{title} > Corpus Statistics", document_title=title
            )
            == "dataset"
        )

    def test_an_unrecognized_heading_stays_other(self, extractor):
        assert extractor.classify_header_path("Latent Dirichlet Allocation") == "other"


# ==============================================================================
# Confidence
# ==============================================================================

class TestResearchPaperConfidence:
    def test_a_paper_with_no_author_line_is_still_a_paper(self, extractor):
        """
        Authors used to be worth 0.15, which put every author-less paper below
        the 0.5 threshold once author extraction stopped inventing them. Below
        the threshold no chunk gets enriched, so every chunk loses its
        section_type -- a metadata fix that silently disables all metadata.
        """
        meta = extractor.extract_from_markdown(NO_AUTHOR_LINE)
        assert meta.authors == []
        assert meta.is_research_paper(), f"confidence was {meta.confidence}"

    def test_a_generic_document_is_not_a_paper(self, extractor):
        markdown = (
            "# Weekly Meeting Notes\n\n"
            "## Attendees\nAlice, Bob\n\n"
            "## Decisions\nShip on Friday.\n\n"
            "## Action Items\nUpdate the changelog.\n"
        )
        meta = extractor.extract_from_markdown(markdown)
        assert not meta.is_research_paper()


# ==============================================================================
# Section position lookup
# ==============================================================================

def test_the_first_section_of_a_document_is_findable(extractor):
    """
    get_section_type truth-tested start_char, so the section beginning at
    character 0 -- every document has one -- was skipped and its chunks came
    back "unknown".
    """
    meta = extractor.extract_from_markdown(NO_AUTHOR_LINE)
    assert meta.sections
    assert meta.sections[0].start_char == 0
    assert meta.get_section_type(0) != "unknown"
