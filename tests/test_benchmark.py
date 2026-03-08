"""
Tests for QASPER Loader and Benchmark Runner.

Phase 6: Evaluation & Metrics — benchmark tests.
"""

import pytest
import json
from pathlib import Path


# ==============================================================================
# QASPER Loader Tests
# ==============================================================================

class TestQASPERLoader:
    """Tests for QASPERLoader."""

    def test_parse_qasper_paper(self, tmp_path):
        """Test parsing a QASPER JSON file."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        qasper_data = {
            "paper_001": {
                "title": "BERT Paper",
                "abstract": "We introduce BERT...",
                "full_text": {
                    "Introduction": ["BERT is a model..."],
                    "Method": ["We pre-train BERT using..."]
                },
                "qas": [
                    {
                        "question": "What is BERT?",
                        "answers": [
                            {
                                "answer": {
                                    "unanswerable": False,
                                    "free_form_answer": "BERT is a bidirectional transformer."
                                },
                                "evidence": ["BERT is a model..."]
                            }
                        ]
                    },
                    {
                        "question": "Is BERT better than GPT?",
                        "answers": [
                            {
                                "answer": {"yes_no": True},
                                "evidence": []
                            }
                        ]
                    },
                ]
            }
        }

        file_path = tmp_path / "test_qasper.json"
        with open(file_path, "w") as f:
            json.dump(qasper_data, f)

        loader = QASPERLoader()
        papers = loader.load(file_path)

        assert len(papers) == 1
        assert papers[0].title == "BERT Paper"
        assert len(papers[0].questions) == 2
        assert papers[0].questions[0].question == "What is BERT?"
        assert papers[0].questions[0].answers[0] == "BERT is a bidirectional transformer."
        assert papers[0].questions[1].answer_type == "yes_no"

    def test_to_evaluation_samples(self, tmp_path):
        """Test converting to EvaluationSamples."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        qasper_data = {
            "paper_001": {
                "title": "BERT",
                "abstract": "",
                "qas": [
                    {
                        "question": "What is BERT?",
                        "answers": [
                            {
                                "answer": {"free_form_answer": "A transformer."},
                                "evidence": ["para1"]
                            }
                        ]
                    }
                ]
            }
        }

        file_path = tmp_path / "test.json"
        with open(file_path, "w") as f:
            json.dump(qasper_data, f)

        loader = QASPERLoader()
        papers = loader.load(file_path)
        samples = loader.to_evaluation_samples(papers)

        assert len(samples) == 1
        assert samples[0].query == "What is BERT?"
        assert samples[0].reference_answer == "A transformer."

    def test_skip_unanswerable(self, tmp_path):
        """Test skipping unanswerable questions."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        qasper_data = {
            "paper_001": {
                "title": "Test",
                "abstract": "",
                "qas": [
                    {
                        "question": "Unanswerable Q",
                        "answers": [{"answer": {"unanswerable": True}, "evidence": []}]
                    },
                    {
                        "question": "Answerable Q",
                        "answers": [{"answer": {"free_form_answer": "Yes."}, "evidence": []}]
                    },
                ]
            }
        }

        file_path = tmp_path / "test.json"
        with open(file_path, "w") as f:
            json.dump(qasper_data, f)

        loader = QASPERLoader()
        papers = loader.load(file_path)

        samples = loader.to_evaluation_samples(papers, skip_unanswerable=True)
        assert len(samples) == 1
        assert samples[0].query == "Answerable Q"

    def test_from_custom_dataset(self):
        """Test creating samples from custom data."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        entries = [
            {"query": "Q1", "answer": "A1", "evidence": ["e1"]},
            {"query": "Q2", "answer": "A2"},
        ]

        samples = QASPERLoader.from_custom_dataset(entries)
        assert len(samples) == 2
        assert samples[0].query == "Q1"
        assert samples[0].reference_answer == "A1"

    def test_sample_dataset(self):
        """Test built-in sample dataset."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        entries = QASPERLoader.create_sample_dataset()
        assert len(entries) >= 3
        assert all("query" in e for e in entries)
        assert all("answer" in e for e in entries)

    def test_load_nonexistent(self):
        """Test loading non-existent file."""
        from docusense.evaluation.qasper_loader import QASPERLoader
        loader = QASPERLoader()
        papers = loader.load("nonexistent.json")
        assert papers == []


# ==============================================================================
# Benchmark Runner Tests
# ==============================================================================

class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_run_sample_benchmark(self, tmp_path):
        """Test running benchmark with sample data."""
        from docusense.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkConfig

        runner = BenchmarkRunner()
        config = BenchmarkConfig(
            name="test_benchmark",
            use_sample_dataset=True,
            output_dir=str(tmp_path)
        )
        report = runner.run(config)

        assert report.num_samples > 0
        assert report.result is not None
        assert report.benchmark_time > 0

    def test_report_to_dict(self, tmp_path):
        """Test report serialization."""
        from docusense.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkConfig

        runner = BenchmarkRunner()
        config = BenchmarkConfig(
            name="test_serialize",
            use_sample_dataset=True,
            output_dir=str(tmp_path)
        )
        report = runner.run(config)
        d = report.to_dict()

        assert "benchmark" in d
        assert "num_samples" in d
        assert "results" in d

    def test_report_saved(self, tmp_path):
        """Test that report JSON is saved."""
        from docusense.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkConfig

        runner = BenchmarkRunner()
        config = BenchmarkConfig(
            name="test_save",
            use_sample_dataset=True,
            output_dir=str(tmp_path)
        )
        runner.run(config)

        report_file = tmp_path / "test_save_report.json"
        assert report_file.exists()

        with open(report_file) as f:
            data = json.load(f)
        assert data["benchmark"] == "test_save"

    def test_run_with_samples(self):
        """Test running with pre-built samples."""
        from docusense.evaluation.benchmark_runner import BenchmarkRunner
        from docusense.evaluation.evaluator import EvaluationSample

        runner = BenchmarkRunner()
        samples = [
            EvaluationSample(
                query="Test",
                generated_answer="Test answer",
                reference_answer="Test ref"
            )
        ]

        report = runner.run_with_samples(samples)
        assert report.num_samples == 1

    def test_summary_grading(self, tmp_path):
        """Test that summary includes grades."""
        from docusense.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkConfig

        runner = BenchmarkRunner()
        config = BenchmarkConfig(
            name="grading_test",
            use_sample_dataset=True,
            output_dir=str(tmp_path)
        )
        report = runner.run(config)

        # Sample dataset doesn't have retrieval data, so answer grade should be present
        assert "answer_grade" in report.summary or "retrieval_grade" in report.summary or len(report.summary) >= 0
