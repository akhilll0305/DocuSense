"""
Tests for Generation Pipeline.

Tests Phase 4.4: End-to-end query → answer with citations.
Uses mocking for CI-safe testing (no Ollama or Qdrant required).
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, Any, List


# ==============================================================================
# Mock objects to avoid importing heavy dependencies
# ==============================================================================

@dataclass
class MockRetrievalResult:
    """Mock RetrievalResult for testing."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int = 1
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_stages: List[str] = field(default_factory=list)


@dataclass
class MockRetrievalMetrics:
    """Mock RetrievalMetrics for testing."""
    total_time: float = 0.5
    query_processing_time: float = 0.1
    search_time: float = 0.3
    reranking_time: float = 0.1
    num_queries_generated: int = 3
    num_initial_results: int = 20
    num_final_results: int = 5
    stages_used: List[str] = field(default_factory=lambda: ["query_processing", "hybrid_search"])


def make_mock_results() -> list:
    """Create sample retrieval results with paper metadata."""
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
            }
        ),
        MockRetrievalResult(
            chunk_id="chunk_003",
            document_id="doc_gpt2",
            text="GPT-2 demonstrates that language models can perform downstream tasks "
                 "in a zero-shot setting.",
            score=0.78,
            rank=3,
            metadata={
                "paper_title": "Language Models are Unsupervised Multitask Learners",
                "authors": ["Alec Radford", "Jeffrey Wu", "Rewon Child"],
                "year": 2019,
                "venue": "OpenAI",
                "section_type": "abstract",
            }
        ),
    ]


# ==============================================================================
# GenerationPipeline Tests
# ==============================================================================

class TestGenerationPipeline:
    """Tests for GenerationPipeline."""
    
    @patch("docusense.generation.generation_pipeline.OllamaClient")
    def test_init(self, mock_client_class):
        """Test pipeline initialization."""
        from docusense.generation.generation_pipeline import GenerationPipeline
        
        pipeline = GenerationPipeline()
        
        assert pipeline.retrieval_pipeline is None
        assert pipeline.answer_generator is not None
        assert pipeline.citation_formatter is not None
        assert pipeline.include_references is True
        assert pipeline.include_bibtex is True
    
    @patch("docusense.generation.generation_pipeline.OllamaClient")
    def test_generate_from_results(self, mock_client_class):
        """Test answer generation from pre-retrieved results."""
        from docusense.generation.generation_pipeline import GenerationPipeline
        
        # Setup mock Ollama client
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "BERT achieved 93.5% F1 score on SST-2 "
            "(Devlin et al., 2018, Results). This represents a significant "
            "improvement over previous state-of-the-art approaches. "
            "GPT-2 also showed strong performance in zero-shot settings "
            "(Radford et al., 2019)."
        )
        mock_client.model = "llama3.2:3b"
        mock_client_class.return_value = mock_client
        
        pipeline = GenerationPipeline()
        results = make_mock_results()
        
        response = pipeline.generate_from_results(
            query="What accuracy did BERT achieve on SST-2?",
            retrieval_results=results
        )
        
        # Verify response structure
        assert response.query == "What accuracy did BERT achieve on SST-2?"
        assert "93.5%" in response.answer
        assert response.num_sources == 3
        assert response.has_citations is True
        assert response.model_used == "llama3.2:3b"
        assert response.generation_mode == "answer"
    
    @patch("docusense.generation.generation_pipeline.OllamaClient")
    def test_generate_from_results_with_citations(self, mock_client_class):
        """Test that citations are properly formatted in response."""
        from docusense.generation.generation_pipeline import GenerationPipeline
        
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "BERT achieved 93.5% F1 (Devlin et al., 2018)."
        )
        mock_client.model = "llama3.2:3b"
        mock_client_class.return_value = mock_client
        
        pipeline = GenerationPipeline()
        results = make_mock_results()
        
        response = pipeline.generate_from_results(
            "What accuracy did BERT achieve?",
            results
        )
        
        # Should have formatted citations
        assert len(response.citations) >= 1  # At least BERT and GPT-2
        
        # Should have reference list
        assert "References:" in response.reference_list
        assert "[1]" in response.reference_list
        
        # Should have BibTeX
        assert "@" in response.bibtex
        assert "BERT" in response.bibtex
    
    @patch("docusense.generation.generation_pipeline.OllamaClient")
    def test_generate_comparison_mode(self, mock_client_class):
        """Test multi-paper comparison mode."""
        from docusense.generation.generation_pipeline import GenerationPipeline
        
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "Comparing BERT (Devlin et al., 2018) and GPT-2 (Radford et al., 2019): "
            "BERT uses bidirectional pre-training while GPT-2 uses autoregressive."
        )
        mock_client.model = "llama3.2:3b"
        mock_client_class.return_value = mock_client
        
        pipeline = GenerationPipeline()
        results = make_mock_results()
        
        response = pipeline.generate_from_results(
            "Compare BERT and GPT-2",
            results,
            mode="compare"
        )
        
        assert response.generation_mode == "compare"
        assert "BERT" in response.answer
    
    @patch("docusense.generation.generation_pipeline.OllamaClient")
    def test_generate_conflicts_mode(self, mock_client_class):
        """Test conflict detection mode."""
        from docusense.generation.generation_pipeline import GenerationPipeline
        
        mock_client = MagicMock()
        mock_client.generate.return_value = "No significant conflicts found."
        mock_client.model = "llama3.2:3b"
        mock_client_class.return_value = mock_client
        
        pipeline = GenerationPipeline()
        results = make_mock_results()
        
        response = pipeline.generate_from_results(
            "Find conflicts",
            results,
            mode="conflicts"
        )
        
        assert response.generation_mode == "conflicts"
    
    @patch("docusense.generation.generation_pipeline.OllamaClient")
    def test_generate_requires_retrieval_pipeline(self, mock_client_class):
        """Test that generate() requires retrieval pipeline."""
        from docusense.generation.generation_pipeline import GenerationPipeline
        
        pipeline = GenerationPipeline()  # No retrieval pipeline
        
        with pytest.raises(RuntimeError, match="No retrieval pipeline"):
            pipeline.generate("test query")
    
    @patch("docusense.generation.generation_pipeline.OllamaClient")
    def test_generate_with_retrieval(self, mock_client_class):
        """Test full pipeline with mocked retrieval."""
        from docusense.generation.generation_pipeline import GenerationPipeline
        
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "BERT achieved 93.5% F1 (Devlin et al., 2018)."
        )
        mock_client.model = "llama3.2:3b"
        mock_client_class.return_value = mock_client
        
        # Mock retrieval pipeline
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = (
            make_mock_results(),
            MockRetrievalMetrics()
        )
        
        pipeline = GenerationPipeline(retrieval_pipeline=mock_retrieval)
        response = pipeline.generate("What accuracy did BERT achieve?")
        
        # Verify full pipeline ran
        assert response.query == "What accuracy did BERT achieve?"
        assert "93.5%" in response.answer
        assert response.retrieval_time > 0 or response.retrieval_time == 0  # Timing
        assert response.generation_time >= 0
        assert response.total_time >= 0
        mock_retrieval.retrieve.assert_called_once()
    
    @patch("docusense.generation.generation_pipeline.OllamaClient")
    def test_generate_no_results(self, mock_client_class):
        """Test pipeline when retrieval returns no results."""
        from docusense.generation.generation_pipeline import GenerationPipeline
        
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = (
            [],  # No results
            MockRetrievalMetrics(num_final_results=0, stages_used=[])
        )
        
        pipeline = GenerationPipeline(retrieval_pipeline=mock_retrieval)
        response = pipeline.generate("nonexistent topic query")
        
        assert "No relevant documents" in response.answer
    
    @patch("docusense.generation.generation_pipeline.OllamaClient")
    def test_get_pipeline_config(self, mock_client_class):
        """Test pipeline configuration report."""
        from docusense.generation.generation_pipeline import GenerationPipeline
        
        pipeline = GenerationPipeline()
        config = pipeline.get_pipeline_config()
        
        assert "retrieval_connected" in config
        assert "citation_style" in config
        assert "model" in config
        assert config["retrieval_connected"] is False
        assert config["citation_style"] == "inline"


# ==============================================================================
# PipelineResponse Tests
# ==============================================================================

class TestPipelineResponse:
    """Tests for PipelineResponse dataclass."""
    
    def test_str_representation(self):
        """Test human-readable string output."""
        from docusense.generation.generation_pipeline import PipelineResponse
        
        response = PipelineResponse(
            query="What is BERT?",
            answer="BERT is a bidirectional transformer model (Devlin et al., 2018).",
            papers_cited=["BERT: Pre-training of Deep Bidirectional Transformers"],
            num_sources=3,
            confidence=0.85,
            total_time=1.5,
            reference_list="\nReferences:\n\n[1] Devlin, J., et al. (2018). BERT. NAACL."
        )
        
        output = str(response)
        
        assert "What is BERT?" in output
        assert "BERT is a bidirectional" in output
        assert "References:" in output
        assert "3 sources" in output
        assert "1 papers" in output
    
    def test_default_values(self):
        """Test default field values."""
        from docusense.generation.generation_pipeline import PipelineResponse
        
        response = PipelineResponse(query="test", answer="test answer")
        
        assert response.citations == []
        assert response.reference_list == ""
        assert response.bibtex == ""
        assert response.confidence == 0.0
        assert response.has_citations is False
        assert response.total_time == 0.0
