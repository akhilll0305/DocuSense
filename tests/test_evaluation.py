"""
Tests for Evaluation Metrics.

Phase 6: Tests for retrieval metrics, answer metrics, and RAG evaluator.
All built-in — no ROUGE or BERTScore libraries required.
"""

import pytest


# ==============================================================================
# Retrieval Metrics Tests
# ==============================================================================

class TestRetrievalMetrics:
    """Tests for RetrievalMetrics."""

    def test_reciprocal_rank_first(self):
        """First result is relevant → MRR = 1.0."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        assert m.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_reciprocal_rank_second(self):
        """Second result is relevant → MRR = 0.5."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        assert m.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5

    def test_reciprocal_rank_none(self):
        """No relevant results → MRR = 0.0."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        assert m.reciprocal_rank(["a", "b", "c"], {"x"}) == 0.0

    def test_precision_at_k(self):
        """Precision@3 with 2 relevant out of 3."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        assert m.precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 3) == pytest.approx(2/3)

    def test_precision_at_1(self):
        """Precision@1 with relevant first result."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        assert m.precision_at_k(["a", "b"], {"a"}, 1) == 1.0

    def test_recall_at_k(self):
        """Recall@3 finding 2 of 3 relevant docs."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        assert m.recall_at_k(["a", "b", "c"], {"a", "b", "d"}, 3) == pytest.approx(2/3)

    def test_recall_at_k_all_found(self):
        """Recall@5 finding all relevant docs."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        assert m.recall_at_k(["a", "b", "c", "d", "e"], {"a", "c"}, 5) == 1.0

    def test_ndcg_at_k_perfect(self):
        """Perfect NDCG@3 when all relevant are at top."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        score = m.ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3)
        assert score == pytest.approx(1.0)

    def test_ndcg_at_k_imperfect(self):
        """NDCG@3 < 1 when relevant docs are not at top."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        score = m.ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3)
        assert score < 1.0
        assert score > 0.0

    def test_ndcg_at_k_none_relevant(self):
        """NDCG = 0 when no relevant docs in results."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        assert m.ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_average_precision(self):
        """Test Average Precision computation."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        # Relevant at positions 1, 3 (0-indexed)
        ap = m.average_precision(["a", "x", "b", "y"], {"a", "b"})
        # AP = (1/1 + 2/3) / 2 = 0.8333
        assert ap == pytest.approx(5/6, abs=0.001)

    def test_evaluate_batch(self):
        """Test batch evaluation averaging."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()

        evals = [
            {"retrieved": ["a", "b", "c"], "relevant": {"a"}},  # MRR = 1.0
            {"retrieved": ["x", "a", "c"], "relevant": {"a"}},  # MRR = 0.5
        ]

        result = m.evaluate_batch(evals)
        assert result.mrr == pytest.approx(0.75)
        assert result.num_queries == 2

    def test_metrics_result_to_dict(self):
        """Test result serialization."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetricsResult
        result = RetrievalMetricsResult(mrr=0.8, ndcg_at_5=0.7)
        d = result.to_dict()
        assert d["MRR"] == 0.8
        assert d["NDCG@5"] == 0.7

    def test_edge_case_empty(self):
        """Test with empty inputs."""
        from docusense.evaluation.retrieval_metrics import RetrievalMetrics
        m = RetrievalMetrics()
        assert m.precision_at_k([], set(), 5) == 0.0
        assert m.recall_at_k([], set(), 5) == 0.0
        assert m.ndcg_at_k([], set(), 5) == 0.0


# ==============================================================================
# Answer Metrics Tests
# ==============================================================================

class TestAnswerMetrics:
    """Tests for AnswerMetrics."""

    def test_token_overlap_identical(self):
        """Identical texts → overlap = 1.0."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        score = m._compute_token_overlap(
            "BERT achieved high accuracy",
            "BERT achieved high accuracy"
        )
        assert score == 1.0

    def test_token_overlap_partial(self):
        """Partial overlap → 0 < score < 1."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        score = m._compute_token_overlap(
            "BERT achieved 93.5% F1",
            "BERT scored 93.5% on SST-2"
        )
        assert 0.0 < score < 1.0

    def test_token_overlap_none(self):
        """No overlap → score = 0."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        score = m._compute_token_overlap("hello", "goodbye")
        assert score == 0.0

    def test_extract_citations_et_al(self):
        """Extract 'et al.' style citations."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        text = "BERT achieved 93.5% (Devlin et al., 2018) on SST-2."
        cited = m._extract_citations(text)
        assert "Devlin" in cited

    def test_extract_citations_multiple(self):
        """Extract multiple citations."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        text = "Comparing BERT (Devlin et al., 2018) and GPT (Radford et al., 2019)."
        cited = m._extract_citations(text)
        assert "Devlin" in cited
        assert "Radford" in cited

    def test_citation_accuracy(self):
        """Test citation accuracy matching."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        result = m._citation_accuracy(
            cited_authors=["Devlin", "Radford"],
            source_papers=["BERT by Devlin et al.", "GPT by Radford et al."]
        )
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_citation_accuracy_partial(self):
        """Cite one author that's not in sources."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        result = m._citation_accuracy(
            cited_authors=["Devlin", "Unknown"],
            source_papers=["BERT by Devlin et al."]
        )
        assert result["precision"] == 0.5

    def test_completeness(self):
        """Test query completeness check."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        score = m._compute_completeness(
            "BERT achieved 93.5% F1 score on sentiment classification",
            "What F1 score did BERT achieve on SST-2?"
        )
        # Should find "F1", "score", "BERT", "achieve" — partial coverage
        assert score > 0.3

    def test_evaluate_full(self):
        """Test full answer evaluation."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        result = m.evaluate(
            generated="BERT achieved 93.5% F1 (Devlin et al., 2018).",
            reference="BERT scored 93.5% F1 on SST-2.",
            query="What F1 did BERT achieve?",
            source_papers=["BERT by Devlin"]
        )

        assert result.token_overlap > 0.0
        assert result.has_citations is True
        assert result.num_citations_found >= 1
        assert result.completeness > 0.0
        assert result.answer_length > 0

    def test_evaluate_batch(self):
        """Test batch evaluation averaging."""
        from docusense.evaluation.answer_metrics import AnswerMetrics
        m = AnswerMetrics()
        evals = [
            {
                "generated": "BERT achieved 93.5% (Devlin et al., 2018).",
                "reference": "BERT scored 93.5%.",
                "query": "What F1 did BERT achieve?",
                "source_papers": ["BERT by Devlin"]
            },
            {
                "generated": "GPT-2 uses autoregressive training (Radford et al., 2019).",
                "reference": "GPT-2 is trained autoregressively.",
                "query": "How is GPT-2 trained?",
                "source_papers": ["GPT-2 by Radford"]
            },
        ]
        result = m.evaluate_batch(evals)
        assert result.token_overlap > 0.0


# ==============================================================================
# RAG Evaluator Tests
# ==============================================================================

class TestRAGEvaluator:
    """Tests for RAGEvaluator."""

    def test_evaluate_full(self):
        """Test end-to-end evaluation."""
        from docusense.evaluation.evaluator import RAGEvaluator, EvaluationSample

        evaluator = RAGEvaluator()

        samples = [
            EvaluationSample(
                query="What F1 did BERT achieve on SST-2?",
                generated_answer="BERT achieved 93.5% F1 (Devlin et al., 2018).",
                reference_answer="BERT scored 93.5% F1 on SST-2.",
                retrieved_ids=["chunk_1", "chunk_2", "chunk_3"],
                relevant_ids=["chunk_1", "chunk_3"],
                source_papers=["BERT by Devlin"]
            ),
            EvaluationSample(
                query="How is GPT-2 trained?",
                generated_answer="GPT-2 uses autoregressive training (Radford et al., 2019).",
                reference_answer="GPT-2 is trained autoregressively.",
                retrieved_ids=["chunk_a", "chunk_b"],
                relevant_ids=["chunk_a"],
                source_papers=["GPT-2 by Radford"]
            ),
        ]

        result = evaluator.evaluate(samples)

        assert result.num_samples == 2
        assert result.retrieval is not None
        assert result.answer is not None
        assert result.retrieval.mrr > 0
        assert result.answer.token_overlap > 0

    def test_evaluate_retrieval_only(self):
        """Test evaluation with retrieval metrics only."""
        from docusense.evaluation.evaluator import RAGEvaluator, EvaluationSample

        evaluator = RAGEvaluator()
        samples = [
            EvaluationSample(
                query="test",
                retrieved_ids=["a", "b"],
                relevant_ids=["a"]
            )
        ]

        result = evaluator.evaluate(samples, evaluate_answers=False)
        assert result.retrieval is not None
        assert result.answer is None

    def test_evaluate_answers_only(self):
        """Test evaluation with answer metrics only."""
        from docusense.evaluation.evaluator import RAGEvaluator, EvaluationSample

        evaluator = RAGEvaluator()
        samples = [
            EvaluationSample(
                query="What is BERT?",
                generated_answer="BERT is a transformer model.",
                reference_answer="BERT is a bidirectional transformer."
            )
        ]

        result = evaluator.evaluate(samples, evaluate_retrieval=False)
        assert result.retrieval is None
        assert result.answer is not None

    def test_evaluation_result_to_dict(self):
        """Test serialization."""
        from docusense.evaluation.evaluator import RAGEvaluator, EvaluationSample

        evaluator = RAGEvaluator()
        samples = [
            EvaluationSample(
                query="test",
                generated_answer="answer",
                reference_answer="ref answer",
                retrieved_ids=["a"],
                relevant_ids=["a"]
            )
        ]

        result = evaluator.evaluate(samples)
        d = result.to_dict()
        assert "retrieval_metrics" in d
        assert "answer_metrics" in d
        assert "num_samples" in d

    def test_save_report(self, tmp_path):
        """Test saving report to JSON."""
        from docusense.evaluation.evaluator import RAGEvaluator, EvaluationResult

        evaluator = RAGEvaluator()
        result = EvaluationResult(num_samples=1, evaluation_time=0.5)

        report_path = tmp_path / "report.json"
        evaluator.save_report(result, report_path)

        assert report_path.exists()
        import json
        with open(report_path) as f:
            data = json.load(f)
        assert data["num_samples"] == 1
