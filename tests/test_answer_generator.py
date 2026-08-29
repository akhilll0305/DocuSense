"""
Tests for Answer Generator and Ollama Client.

Tests Phase 4.1: Ollama integration and answer generation.
Uses mocking for CI-safe testing (no Ollama required).
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, Any


# ==============================================================================
# Mock RetrievalResult for testing (avoid importing heavy dependencies)
# ==============================================================================

@dataclass
class MockRetrievalResult:
    """Mock RetrievalResult for testing without full retrieval stack."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


def make_mock_results() -> list:
    """Create sample RetrievalResult objects with paper metadata."""
    return [
        MockRetrievalResult(
            chunk_id="chunk_001",
            document_id="doc_bert",
            text="BERT achieved 93.5% F1 score on the SST-2 sentiment classification benchmark, "
                 "surpassing previous state-of-the-art results by 5.3 percentage points.",
            score=0.92,
            rank=1,
            metadata={
                "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
                "year": 2018,
                "venue": "NAACL",
                "section_type": "results",
                "has_citations": True,
                "has_equations": False
            }
        ),
        MockRetrievalResult(
            chunk_id="chunk_002",
            document_id="doc_bert",
            text="We pre-train BERT using two unsupervised tasks: Masked Language Modeling (MLM) "
                 "and Next Sentence Prediction (NSP).",
            score=0.85,
            rank=2,
            metadata={
                "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
                "year": 2018,
                "venue": "NAACL",
                "section_type": "methodology",
                "has_citations": False,
                "has_equations": True
            }
        ),
        MockRetrievalResult(
            chunk_id="chunk_003",
            document_id="doc_gpt2",
            text="GPT-2 demonstrates that language models can perform downstream tasks in a "
                 "zero-shot setting without any parameter or architecture modification.",
            score=0.78,
            rank=3,
            metadata={
                "paper_title": "Language Models are Unsupervised Multitask Learners",
                "authors": ["Alec Radford", "Jeffrey Wu", "Rewon Child"],
                "year": 2019,
                "venue": "OpenAI",
                "section_type": "abstract",
                "has_citations": True,
                "has_equations": False
            }
        ),
    ]


# ==============================================================================
# OllamaClient Tests
# ==============================================================================

class TestOllamaClient:
    """Tests for OllamaClient."""
    
    @patch("docusense.llms.ollama_client.OLLAMA_AVAILABLE", True)
    @patch("docusense.llms.ollama_client.ollama", create=True)
    def test_init(self, mock_ollama):
        """Test OllamaClient initialization."""
        from docusense.llms.ollama_client import OllamaClient
        
        client = OllamaClient(
            model="llama3.2:3b",
            base_url="http://localhost:11434"
        )
        
        assert client.model == "llama3.2:3b"
        assert client.base_url == "http://localhost:11434"
    
    @patch("docusense.llms.ollama_client.OLLAMA_AVAILABLE", True)
    @patch("docusense.llms.ollama_client.ollama", create=True)
    def test_generate(self, mock_ollama):
        """Test text generation."""
        from docusense.llms.ollama_client import OllamaClient
        
        # Mock the client and response
        mock_client_instance = MagicMock()
        mock_ollama.Client.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.response = "BERT is a bidirectional transformer model."
        mock_client_instance.generate.return_value = mock_response
        
        client = OllamaClient()
        result = client.generate("What is BERT?")
        
        assert result == "BERT is a bidirectional transformer model."
        mock_client_instance.generate.assert_called_once()
    
    @patch("docusense.llms.ollama_client.OLLAMA_AVAILABLE", True)
    @patch("docusense.llms.ollama_client.ollama", create=True)
    def test_chat(self, mock_ollama):
        """Test chat-based generation."""
        from docusense.llms.ollama_client import OllamaClient
        
        mock_client_instance = MagicMock()
        mock_ollama.Client.return_value = mock_client_instance
        
        mock_message = MagicMock()
        mock_message.content = "BERT uses masked language modeling."
        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_client_instance.chat.return_value = mock_response
        
        client = OllamaClient()
        messages = [
            {"role": "user", "content": "How does BERT work?"}
        ]
        result = client.chat(messages)
        
        assert result == "BERT uses masked language modeling."
    
    @patch("docusense.llms.ollama_client.OLLAMA_AVAILABLE", True)
    @patch("docusense.llms.ollama_client.ollama", create=True)
    def test_is_available_with_model(self, mock_ollama):
        """Test model availability check."""
        from docusense.llms.ollama_client import OllamaClient
        
        mock_client_instance = MagicMock()
        mock_ollama.Client.return_value = mock_client_instance
        
        mock_model = MagicMock()
        mock_model.model = "llama3.2:3b"
        mock_models = MagicMock()
        mock_models.models = [mock_model]
        mock_client_instance.list.return_value = mock_models
        
        client = OllamaClient(model="llama3.2:3b")
        assert client.is_available() is True
    
    @patch("docusense.llms.ollama_client.OLLAMA_AVAILABLE", False)
    def test_not_available_without_package(self):
        """Test graceful handling when ollama package not installed."""
        from docusense.llms.ollama_client import OllamaClient
        
        client = OllamaClient()
        assert client.is_available() is False
    
    @patch("docusense.llms.ollama_client.OLLAMA_AVAILABLE", False)
    def test_generate_raises_when_unavailable(self):
        """Test that generate raises RuntimeError when Ollama unavailable."""
        from docusense.llms.ollama_client import OllamaClient
        
        client = OllamaClient()
        with pytest.raises(RuntimeError, match="Ollama is not available"):
            client.generate("test prompt")


# ==============================================================================
# AnswerGenerator Tests
# ==============================================================================

class TestAnswerGenerator:
    """Tests for AnswerGenerator."""
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_init(self, mock_client_class):
        """Test AnswerGenerator initialization."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        generator = AnswerGenerator()
        assert generator.client is not None
        assert generator.include_citations is True
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_build_context(self, mock_client_class):
        """Test context building from retrieval results."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        generator = AnswerGenerator()
        results = make_mock_results()
        
        context, sources = generator._build_context(results)
        
        # Context should contain paper metadata
        assert "BERT" in context
        assert "Source 1" in context or "[Source 1]" in context
        assert len(sources) == 3
        
        # Sources should have paper metadata
        assert sources[0]["paper_title"] == "BERT: Pre-training of Deep Bidirectional Transformers"
        assert sources[0]["year"] == 2018
        assert sources[0]["section_type"] == "results"
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_build_context_by_paper(self, mock_client_class):
        """Test context grouping by paper."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        generator = AnswerGenerator()
        results = make_mock_results()
        
        context, sources = generator._build_context_by_paper(results)
        
        # Should have sections from both papers
        assert "BERT" in context
        assert "Language Models are Unsupervised" in context
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_generate_answer(self, mock_client_class):
        """Test answer generation with mocked LLM."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        # Setup mock
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "BERT achieved 93.5% F1 score on SST-2 "
            "(Devlin et al., 2018, Results). This represents a significant "
            "improvement over previous approaches."
        )
        mock_client.model = "llama3.2:3b"
        mock_client_class.return_value = mock_client
        
        generator = AnswerGenerator()
        results = make_mock_results()
        
        answer = generator.generate_answer(
            "What accuracy did BERT achieve on SST-2?",
            results
        )
        
        # Verify answer structure
        assert answer.query == "What accuracy did BERT achieve on SST-2?"
        assert "93.5%" in answer.answer
        assert answer.num_sources == 3
        assert answer.has_citations is True  # Contains (Devlin et al., 2018)
        assert answer.model_used == "llama3.2:3b"
        assert len(answer.papers_cited) == 2  # BERT and GPT-2 papers
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_compare_papers(self, mock_client_class):
        """Test multi-paper comparison."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "BERT uses bidirectional pre-training (Devlin et al., 2018), "
            "while GPT-2 uses unidirectional autoregressive training "
            "(Radford et al., 2019)."
        )
        mock_client.model = "llama3.2:3b"
        mock_client_class.return_value = mock_client
        
        generator = AnswerGenerator()
        results = make_mock_results()
        
        answer = generator.compare_papers(
            "Compare BERT and GPT-2 training approaches",
            results
        )
        
        assert answer.is_multi_paper is True
        assert len(answer.papers_cited) >= 1
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_format_author_string(self, mock_client_class):
        """Test author formatting for citations."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        # Single author
        assert AnswerGenerator._format_author_string(["John Smith"]) == "John Smith"
        
        # Two authors
        assert AnswerGenerator._format_author_string(
            ["John Smith", "Jane Doe"]
        ) == "John Smith and Jane Doe"
        
        # 3+ authors (et al.)
        assert AnswerGenerator._format_author_string(
            ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"]
        ) == "Devlin et al."
        
        # Empty
        assert AnswerGenerator._format_author_string([]) == ""
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_estimate_confidence(self, mock_client_class):
        """Test confidence estimation."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        results = make_mock_results()
        
        # Good answer with sources
        conf = AnswerGenerator._estimate_confidence(
            results,
            "A detailed answer about BERT's performance on SST-2 benchmark."
        )
        assert 0.0 < conf <= 1.0
        
        # Empty answer
        conf_empty = AnswerGenerator._estimate_confidence(results, "")
        assert conf_empty == 0.0
        
        # No results
        conf_no_results = AnswerGenerator._estimate_confidence([], "Some answer")
        assert conf_no_results == 0.0
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_check_has_citations(self, mock_client_class):
        """Test citation detection in answer text."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        # With citation
        assert AnswerGenerator._check_has_citations(
            "BERT achieved 93.5% (Devlin et al., 2018)"
        ) is True
        
        # Without citation
        assert AnswerGenerator._check_has_citations(
            "BERT achieved 93.5% accuracy"
        ) is False
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_extract_unique_papers(self, mock_client_class):
        """Test paper deduplication."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        sources = [
            {"paper_title": "Paper A"},
            {"paper_title": "Paper A"},  # Duplicate
            {"paper_title": "Paper B"},
            {"paper_title": "Unknown Document"},  # Should be excluded
        ]
        
        papers = AnswerGenerator._extract_unique_papers(sources)
        assert papers == ["Paper A", "Paper B"]
    
    @patch("docusense.generation.answer_generator.get_llm_client")
    def test_generate_answer_handles_llm_error(self, mock_client_class):
        """Test graceful handling when LLM fails."""
        from docusense.generation.answer_generator import AnswerGenerator
        
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("Ollama not running")
        mock_client.model = "llama3.2:3b"
        mock_client_class.return_value = mock_client
        
        generator = AnswerGenerator()
        results = make_mock_results()
        
        answer = generator.generate_answer("What is BERT?", results)
        
        # Should not raise, but include error message
        assert "unable to generate" in answer.answer.lower() or "technical issue" in answer.answer.lower()


class TestCitationValidation:
    """
    Citations must be traceable to a retrieved source.

    Small local models invent plausible-looking citations even when the prompt
    forbids it, and a fabricated citation is worse than none in a system whose
    whole purpose is grounded attribution. These cover the post-generation
    check that removes unsupported ones.
    """

    @staticmethod
    def _generator():
        from docusense.generation.answer_generator import AnswerGenerator
        # Bypass __init__ so the test needs no Ollama connection.
        return AnswerGenerator.__new__(AnswerGenerator)

    PAPER_SOURCE = [{
        "authors": ["Aicha Saadi", "Noureddine Abghour", "Zouhair Chiba"],
        "year": 2025,
        "section_type": "results",
    }]
    UNATTRIBUTED_SOURCE = [{"authors": [], "year": "n.d.", "section_type": "unknown"}]

    def test_supported_citation_is_kept(self):
        text = "Delay fell by 23% (Saadi et al., 2025, results)."
        out, removed = self._generator()._strip_unsupported_citations(text, self.PAPER_SOURCE)
        assert removed == 0
        assert out == text

    def test_citation_without_section_is_kept(self):
        """The model routinely drops the section suffix; that is still valid."""
        text = "Delay fell by 23% (Saadi et al., 2025)."
        out, removed = self._generator()._strip_unsupported_citations(text, self.PAPER_SOURCE)
        assert removed == 0
        assert "(Saadi et al., 2025)" in out

    def test_alternative_author_rendering_is_kept(self):
        """
        "Saadi et al.", "Saadi and Abghour", and "A. Saadi" all name the same
        source, so validation matches on surname and year, not on an exact
        string. Requiring an exact match deleted legitimate citations.
        """
        for rendering in [
            "(Saadi et al., 2025, results)",
            "(Saadi and Abghour, 2025)",
            "(Abghour et al., 2025)",
        ]:
            text = f"Delay fell by 23% {rendering}."
            out, removed = self._generator()._strip_unsupported_citations(
                text, self.PAPER_SOURCE
            )
            assert removed == 0, f"wrongly removed {rendering}"

    def test_narrative_citation_is_kept(self):
        """"Saadi et al. (2025)" puts the name outside the parentheses."""
        text = "According to Saadi et al. (2025, results), delay fell by 23%."
        out, removed = self._generator()._strip_unsupported_citations(text, self.PAPER_SOURCE)
        assert removed == 0
        assert "Saadi et al. (2025, results)" in out

    def test_fabricated_narrative_citation_is_removed(self):
        """A fabricated narrative citation takes its lead-in phrase with it."""
        text = "According to Smith et al. (2019), results improved."
        out, removed = self._generator()._strip_unsupported_citations(text, self.PAPER_SOURCE)
        assert removed == 1
        assert "Smith" not in out and "According to" not in out
        assert out == "Results improved."

    def test_wrong_year_is_removed(self):
        text = "Delay fell by 23% (Saadi et al., 1999)."
        out, removed = self._generator()._strip_unsupported_citations(text, self.PAPER_SOURCE)
        assert removed == 1
        assert "1999" not in out

    def test_wrong_author_is_removed(self):
        text = "Delay fell by 23% (Smith et al., 2019, results)."
        out, removed = self._generator()._strip_unsupported_citations(text, self.PAPER_SOURCE)
        assert removed == 1
        assert "Smith" not in out

    def test_fabricated_citation_on_unattributed_source_is_removed(self):
        """Regression: a source with no metadata drew an invented citation."""
        text = "SGD at lr 0.01 was used (Saadi et al., 2023, Methods)."
        out, removed = self._generator()._strip_unsupported_citations(
            text, self.UNATTRIBUTED_SOURCE
        )
        assert removed == 1
        assert "Saadi" not in out
        assert out == "SGD at lr 0.01 was used."

    def test_ordinary_parentheses_are_untouched(self):
        text = "The model (a CNN) ran for 90 epochs (roughly two days)."
        out, removed = self._generator()._strip_unsupported_citations(
            text, self.UNATTRIBUTED_SOURCE
        )
        assert removed == 0
        assert out == text

    def test_multiple_fabrications_all_removed(self):
        text = "A holds (Lee et al., 2020). B holds (Kim et al., 2021, methods)."
        out, removed = self._generator()._strip_unsupported_citations(
            text, self.PAPER_SOURCE
        )
        assert removed == 2
        assert "Lee" not in out and "Kim" not in out

    def test_punctuation_survives_removal(self):
        """Removal must not eat the sentence's terminating punctuation."""
        text = "SGD at lr 0.01 was used (Nobody et al., 2001)."
        out, removed = self._generator()._strip_unsupported_citations(
            text, self.UNATTRIBUTED_SOURCE
        )
        assert removed == 1
        assert out.endswith(".")
        assert out == "SGD at lr 0.01 was used."

    def test_has_citations_detects_both_renderings(self):
        from docusense.generation.answer_generator import AnswerGenerator
        assert AnswerGenerator._check_has_citations("Delay fell (Saadi et al., 2025).")
        assert AnswerGenerator._check_has_citations("Saadi et al. (2025) report a drop.")
        assert not AnswerGenerator._check_has_citations("The model (a CNN) ran 90 epochs.")
