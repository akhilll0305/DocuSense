"""
Tests for QASPER Loader, the QASPER harness helpers, and the Benchmark Runner.

The fixtures here deliberately mirror the *released* QASPER v0.3 shape:
`full_text` is a list of {"section_name", "paragraphs"}, and `evidence` sits
inside the `answer` object. An earlier version of these tests used a dict for
`full_text` and put `evidence` beside the answer; they passed while the loader
could not read a single real paper. Fixtures that do not match the real data
are not tests.
"""

import json

import pytest


# A minimal but format-accurate QASPER entry.
REAL_FORMAT_PAPER = {
    "paper_001": {
        "title": "BERT Paper",
        "abstract": "We introduce BERT...",
        "full_text": [
            {
                "section_name": "Introduction",
                "paragraphs": [
                    "BERT is a bidirectional transformer model.",
                    "It is pre-trained on unlabeled text.",
                ],
            },
            {
                "section_name": "Experiments ::: Automatic Evaluation",
                "paragraphs": ["We pre-train BERT using masked language modelling."],
            },
        ],
        "qas": [
            {
                "question": "What is BERT?",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": None,
                            "free_form_answer": "BERT is a bidirectional transformer.",
                            "evidence": ["BERT is a bidirectional transformer model."],
                            "highlighted_evidence": [],
                        },
                        "annotation_id": "a1",
                        "worker_id": "w1",
                    }
                ],
            },
            {
                "question": "Is BERT better than GPT?",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": True,
                            "free_form_answer": "",
                            "evidence": [],
                            "highlighted_evidence": [],
                        },
                        "annotation_id": "a2",
                        "worker_id": "w2",
                    }
                ],
            },
        ],
    }
}


def write_dataset(tmp_path, data, name="qasper.json"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


# ==============================================================================
# QASPER Loader Tests
# ==============================================================================

class TestQASPERLoader:
    """Tests for QASPERLoader."""

    def test_parse_qasper_paper(self, tmp_path):
        """Parse a paper in the released QASPER format."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        path = write_dataset(tmp_path, REAL_FORMAT_PAPER)
        papers = QASPERLoader().load(path)

        assert len(papers) == 1
        assert papers[0].title == "BERT Paper"
        assert len(papers[0].questions) == 2
        assert papers[0].questions[0].question == "What is BERT?"
        assert papers[0].questions[0].answers[0] == "BERT is a bidirectional transformer."
        assert papers[0].questions[1].answer_type == "yes_no"

    def test_full_text_list_form_yields_sections(self, tmp_path):
        """
        `full_text` is a LIST in the real dataset.

        Reading it as a dict silently produced zero sections, which left the
        reconstructed document empty and nothing to retrieve against.
        """
        from docusense.evaluation.qasper_loader import QASPERLoader

        path = write_dataset(tmp_path, REAL_FORMAT_PAPER)
        paper = QASPERLoader().load(path)[0]

        assert len(paper.full_text_sections) == 2
        assert paper.full_text_sections[0]["section"] == "Introduction"
        assert len(paper.paragraphs()) == 3

    def test_full_text_dict_form_still_supported(self, tmp_path):
        """A dict of section -> paragraphs is accepted too."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        data = {
            "p1": {
                "title": "T",
                "abstract": "",
                "full_text": {"Introduction": ["Para one.", "Para two."]},
                "qas": [],
            }
        }
        paper = QASPERLoader().load(write_dataset(tmp_path, data))[0]

        assert len(paper.full_text_sections) == 1
        assert paper.full_text_sections[0]["paragraphs"] == ["Para one.", "Para two."]

    def test_evidence_read_from_inside_answer(self, tmp_path):
        """
        Evidence lives inside the `answer` object.

        Reading it as a sibling of `answer` returned an empty evidence list for
        every real question, which leaves no ground truth to score against.
        """
        from docusense.evaluation.qasper_loader import QASPERLoader

        path = write_dataset(tmp_path, REAL_FORMAT_PAPER)
        paper = QASPERLoader().load(path)[0]

        assert paper.questions[0].evidence == [
            "BERT is a bidirectional transformer model."
        ]

    def test_evidence_legacy_sibling_position(self, tmp_path):
        """Evidence beside the answer (older fixtures) is still read."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        data = {
            "p1": {
                "title": "T",
                "abstract": "",
                "full_text": [],
                "qas": [
                    {
                        "question": "Q?",
                        "answers": [
                            {
                                "answer": {"free_form_answer": "A."},
                                "evidence": ["legacy evidence"],
                            }
                        ],
                    }
                ],
            }
        }
        paper = QASPERLoader().load(write_dataset(tmp_path, data))[0]
        assert paper.questions[0].evidence == ["legacy evidence"]

    def test_evidence_unioned_across_annotators(self, tmp_path):
        """Each annotator's evidence counts, deduplicated, in order."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        data = {
            "p1": {
                "title": "T",
                "abstract": "",
                "full_text": [],
                "qas": [
                    {
                        "question": "Q?",
                        "answers": [
                            {"answer": {"free_form_answer": "A.", "evidence": ["one", "two"]}},
                            {"answer": {"free_form_answer": "B.", "evidence": ["two", "three"]}},
                        ],
                    }
                ],
            }
        }
        paper = QASPERLoader().load(write_dataset(tmp_path, data))[0]
        assert paper.questions[0].evidence == ["one", "two", "three"]

    def test_body_evidence_drops_floats(self, tmp_path):
        """"FLOAT SELECTED" evidence points at a figure, not body text."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        data = {
            "p1": {
                "title": "T",
                "abstract": "",
                "full_text": [],
                "qas": [
                    {
                        "question": "Q?",
                        "answers": [
                            {
                                "answer": {
                                    "free_form_answer": "A.",
                                    "evidence": [
                                        "FLOAT SELECTED: Table 1 shows results.",
                                        "A real paragraph.",
                                    ],
                                }
                            }
                        ],
                    }
                ],
            }
        }
        loader = QASPERLoader()
        question = loader.load(write_dataset(tmp_path, data))[0].questions[0]

        assert len(question.evidence) == 2
        assert loader.body_evidence(question) == ["A real paragraph."]

    def test_unanswerable_only_when_no_annotator_answered(self, tmp_path):
        """One annotator answering is enough to keep a question."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        data = {
            "p1": {
                "title": "T",
                "abstract": "",
                "full_text": [],
                "qas": [
                    {
                        "question": "Partly answered?",
                        "answers": [
                            {"answer": {"unanswerable": True}},
                            {"answer": {"free_form_answer": "Actually yes."}},
                        ],
                    },
                    {
                        "question": "Never answered?",
                        "answers": [{"answer": {"unanswerable": True}}],
                    },
                ],
            }
        }
        paper = QASPERLoader().load(write_dataset(tmp_path, data))[0]

        assert paper.questions[0].is_unanswerable is False
        assert paper.questions[1].is_unanswerable is True

    def test_reconstruct_document(self, tmp_path):
        """The rebuilt Markdown keeps evidence paragraphs verbatim."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        path = write_dataset(tmp_path, REAL_FORMAT_PAPER)
        paper = QASPERLoader().load(path)[0]
        markdown = QASPERLoader.reconstruct_document(paper)

        assert markdown.startswith("# BERT Paper")
        assert "## Abstract" in markdown
        assert "## Introduction" in markdown
        # Nested section paths become deeper headers, named by their leaf.
        assert "### Automatic Evaluation" in markdown
        # Verbatim text is what makes evidence matchable after ingestion.
        assert "BERT is a bidirectional transformer model." in markdown

    def test_to_evaluation_samples(self, tmp_path):
        """Samples carry evidence text, and no synthetic relevant ids."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        path = write_dataset(tmp_path, REAL_FORMAT_PAPER)
        loader = QASPERLoader()
        samples = loader.to_evaluation_samples(loader.load(path))

        assert len(samples) == 2
        assert samples[0].query == "What is BERT?"
        assert samples[0].reference_answer == "BERT is a bidirectional transformer."
        assert samples[0].paper_id == "paper_001"
        assert samples[0].evidence_texts == [
            "BERT is a bidirectional transformer model."
        ]
        # Placeholder ids would score every retrieval metric at exactly zero.
        assert samples[0].relevant_ids == []

    def test_skip_unanswerable(self, tmp_path):
        """Unanswerable questions are skipped by default."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        data = {
            "p1": {
                "title": "Test",
                "abstract": "",
                "full_text": [],
                "qas": [
                    {"question": "Unanswerable Q", "answers": [{"answer": {"unanswerable": True}}]},
                    {"question": "Answerable Q", "answers": [{"answer": {"free_form_answer": "Yes."}}]},
                ],
            }
        }
        loader = QASPERLoader()
        papers = loader.load(write_dataset(tmp_path, data))

        samples = loader.to_evaluation_samples(papers, skip_unanswerable=True)
        assert len(samples) == 1
        assert samples[0].query == "Answerable Q"

    def test_max_samples_caps_output(self, tmp_path):
        """max_samples limits how many questions are returned."""
        from docusense.evaluation.qasper_loader import QASPERLoader

        loader = QASPERLoader()
        papers = loader.load(write_dataset(tmp_path, REAL_FORMAT_PAPER))
        assert len(loader.to_evaluation_samples(papers, max_samples=1)) == 1

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
# Ground-truth matching (QASPER harness helpers)
# ==============================================================================

class TestEvidenceMatching:
    """Tests for the text matching that grounds evidence to chunk ids."""

    def test_normalize_survives_ingestion_rewrites(self):
        """Curly quotes, dashes and reflowed whitespace fold to one form."""
        from docusense.evaluation.qasper_harness import normalize_text

        raw = "The  model’s   state—of—the—art\nresult"
        stored = "The model's state-of-the-art result"
        assert normalize_text(raw) == normalize_text(stored)

    def test_coverage_full_containment(self):
        """A paragraph wholly inside a chunk covers all of itself."""
        from docusense.evaluation.qasper_harness import coverage

        evidence = "we trained the model for ten epochs".split()
        chunk = ("some preamble we trained the model for ten epochs and more text").split()

        evidence_cov, chunk_cov = coverage(evidence, chunk)
        assert evidence_cov == 1.0
        assert 0.0 < chunk_cov < 1.0

    def test_coverage_partial_overlap(self):
        """A straddling paragraph is covered in part by each chunk."""
        from docusense.evaluation.qasper_harness import coverage

        evidence = "alpha beta gamma delta epsilon zeta".split()
        chunk = "gamma delta epsilon zeta".split()

        evidence_cov, chunk_cov = coverage(evidence, chunk)
        assert evidence_cov == pytest.approx(4 / 6)
        assert chunk_cov == 1.0

    def test_coverage_no_overlap(self):
        """Unrelated text shares nothing."""
        from docusense.evaluation.qasper_harness import coverage

        evidence_cov, chunk_cov = coverage("alpha beta".split(), "gamma delta".split())
        assert evidence_cov == 0.0
        assert chunk_cov == 0.0

    def test_coverage_matches_through_common_words(self):
        """
        Long sequences still match through frequent tokens.

        SequenceMatcher's `autojunk` heuristic treats items appearing in more
        than 1% of a >200-item sequence as junk, which for prose means "the",
        "of" and "and" stop matching. It must stay disabled.
        """
        from docusense.evaluation.qasper_harness import coverage

        filler = "the model of the system and the data ".split() * 40  # >200 tokens
        evidence = filler[:120]
        evidence_cov, _ = coverage(evidence, filler)
        assert evidence_cov == 1.0

    def test_empty_inputs(self):
        """Empty input covers nothing rather than raising."""
        from docusense.evaluation.qasper_harness import coverage

        assert coverage([], ["a"]) == (0.0, 0.0)
        assert coverage(["a"], []) == (0.0, 0.0)


class TestArmConfiguration:
    """The ablation arms must isolate one component at a time."""

    def test_arms_add_one_component_at_a_time(self):
        from docusense.evaluation.qasper_harness import DEFAULT_ARMS

        by_name = {a.name: a for a in DEFAULT_ARMS}

        assert by_name["vector"].enable_hybrid_search is False
        assert by_name["vector"].enable_reranking is False
        assert by_name["vector"].enable_query_processing is False

        # hybrid differs from vector only in hybrid search
        assert by_name["hybrid"].enable_hybrid_search is True
        assert by_name["hybrid"].enable_reranking is False
        assert by_name["hybrid"].enable_query_processing is False

        # hybrid_rerank differs from hybrid only in reranking
        assert by_name["hybrid_rerank"].enable_hybrid_search is True
        assert by_name["hybrid_rerank"].enable_reranking is True
        assert by_name["hybrid_rerank"].enable_query_processing is False

        # full differs from hybrid_rerank only in query processing
        assert by_name["full"].enable_hybrid_search is True
        assert by_name["full"].enable_reranking is True
        assert by_name["full"].enable_query_processing is True


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

    def test_unrun_samples_are_reported_not_silently_empty(self, tmp_path):
        """
        A report with no metrics must say why.

        The built-in samples have no retrieved ids and no generated answers, so
        the evaluator skips all of them. Previously that produced an empty
        `results` block indistinguishable from a genuine score of zero.
        """
        from docusense.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkConfig

        runner = BenchmarkRunner()
        report = runner.run(BenchmarkConfig(
            name="warn_test",
            use_sample_dataset=True,
            output_dir=str(tmp_path),
        ))

        assert report.result.retrieval is None
        assert report.warnings, "an unmeasurable report must carry a warning"
        assert any("retrieval" in w.lower() for w in report.warnings)
        assert "warnings" in report.to_dict()

    def test_no_warnings_when_samples_are_scorable(self, tmp_path):
        """Filled samples produce metrics and no warnings."""
        from docusense.evaluation.benchmark_runner import BenchmarkRunner
        from docusense.evaluation.evaluator import EvaluationSample

        samples = [
            EvaluationSample(
                query="Q",
                generated_answer="An answer.",
                reference_answer="An answer.",
                retrieved_ids=["c1", "c2"],
                relevant_ids=["c1"],
            )
        ]
        report = BenchmarkRunner().run_with_samples(samples)

        assert report.result.retrieval is not None
        assert report.result.retrieval.mrr == 1.0

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

    def test_summary_grading(self):
        """
        Grades are derived from the metrics.

        The previous version of this test ended in `or len(report.summary) >= 0`,
        which is true of every possible value and so asserted nothing.
        """
        from docusense.evaluation.benchmark_runner import BenchmarkRunner
        from docusense.evaluation.evaluator import EvaluationSample

        perfect = [
            EvaluationSample(
                query="Q",
                generated_answer="A.",
                reference_answer="A.",
                retrieved_ids=["c1"],
                relevant_ids=["c1"],
            )
        ]
        report = BenchmarkRunner().run_with_samples(perfect)

        assert report.summary["retrieval_grade"] == "Excellent"
        assert report.summary["retrieval_highlights"]["MRR"] == 1.0

        missed = [
            EvaluationSample(
                query="Q",
                generated_answer="A.",
                reference_answer="A.",
                retrieved_ids=["c9"],
                relevant_ids=["c1"],
            )
        ]
        report = BenchmarkRunner().run_with_samples(missed)
        assert report.summary["retrieval_grade"] == "Needs Improvement"
