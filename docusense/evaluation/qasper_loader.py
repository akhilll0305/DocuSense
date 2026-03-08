"""
QASPER Dataset Loader - Load and parse the QASPER benchmark.

Phase 6: Evaluation & Metrics (Step 4)

PURPOSE:
--------
QASPER (Question Answering on Scientific Papers) is the standard
benchmark for evaluating QA on research papers. This loader:
1. Parses QASPER JSON format
2. Extracts question-answer pairs with evidence
3. Converts to EvaluationSample format for our evaluator
4. Supports creating custom evaluation datasets

QASPER Format:
--------------
{
    "paper_id": {
        "title": "...",
        "abstract": "...",
        "full_text": {...},
        "qas": [
            {
                "question": "...",
                "answers": [
                    {
                        "answer": {"unanswerable": false, "free_form_answer": "..."},
                        "evidence": ["paragraph1", "paragraph2"]
                    }
                ]
            }
        ]
    }
}

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from loguru import logger

from docusense.evaluation.evaluator import EvaluationSample


@dataclass
class QASPERQuestion:
    """A single QASPER question with answers and evidence."""
    question: str
    answers: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    is_unanswerable: bool = False
    answer_type: str = "free_form"  # free_form, extractive, yes_no, unanswerable
    paper_id: str = ""
    paper_title: str = ""


@dataclass
class QASPERPaper:
    """A QASPER paper entry with its QA pairs."""
    paper_id: str
    title: str = ""
    abstract: str = ""
    questions: List[QASPERQuestion] = field(default_factory=list)
    full_text_sections: List[Dict[str, Any]] = field(default_factory=list)


class QASPERLoader:
    """
    Load and parse the QASPER benchmark dataset.

    Usage:
        # From QASPER JSON file
        loader = QASPERLoader()
        papers = loader.load("qasper-test-v0.3.json")
        samples = loader.to_evaluation_samples(papers)

        # From custom dataset
        samples = loader.from_custom_dataset([
            {"query": "What is BERT?", "answer": "...", "evidence": [...]}
        ])
    """

    def load(self, file_path: str | Path) -> List[QASPERPaper]:
        """
        Load QASPER dataset from JSON file.

        Args:
            file_path: Path to QASPER JSON file

        Returns:
            List of QASPERPaper objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"QASPER file not found: {file_path}")
            return []

        logger.info(f"📚 Loading QASPER dataset from {file_path.name}...")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        papers = []
        for paper_id, paper_data in data.items():
            paper = self._parse_paper(paper_id, paper_data)
            papers.append(paper)

        total_qs = sum(len(p.questions) for p in papers)
        logger.success(f"✅ Loaded {len(papers)} papers, {total_qs} questions")

        return papers

    def _parse_paper(self, paper_id: str, data: Dict[str, Any]) -> QASPERPaper:
        """Parse a single QASPER paper entry."""
        title = data.get("title", "")
        abstract = data.get("abstract", "")

        # Parse full text sections
        sections = []
        full_text = data.get("full_text", {})
        if isinstance(full_text, dict):
            for sec_name, paragraphs in full_text.items():
                sections.append({
                    "section": sec_name,
                    "paragraphs": paragraphs if isinstance(paragraphs, list) else [str(paragraphs)]
                })

        # Parse questions
        questions = []
        for qa in data.get("qas", []):
            question_text = qa.get("question", "")
            answers = []
            evidence = []
            is_unanswerable = False
            answer_type = "free_form"

            for ans_entry in qa.get("answers", []):
                ans_obj = ans_entry.get("answer", {})

                if ans_obj.get("unanswerable", False):
                    is_unanswerable = True
                    answer_type = "unanswerable"
                elif ans_obj.get("yes_no") is not None:
                    answers.append("Yes" if ans_obj["yes_no"] else "No")
                    answer_type = "yes_no"
                elif ans_obj.get("extractive_spans"):
                    answers.extend(ans_obj["extractive_spans"])
                    answer_type = "extractive"
                elif ans_obj.get("free_form_answer"):
                    answers.append(ans_obj["free_form_answer"])
                    answer_type = "free_form"

                # Collect evidence paragraphs
                for ev in ans_entry.get("evidence", []):
                    if isinstance(ev, str) and ev.strip():
                        evidence.append(ev)

            questions.append(QASPERQuestion(
                question=question_text,
                answers=answers,
                evidence=evidence,
                is_unanswerable=is_unanswerable,
                answer_type=answer_type,
                paper_id=paper_id,
                paper_title=title
            ))

        return QASPERPaper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            questions=questions,
            full_text_sections=sections
        )

    def to_evaluation_samples(
        self,
        papers: List[QASPERPaper],
        skip_unanswerable: bool = True,
        max_samples: Optional[int] = None
    ) -> List[EvaluationSample]:
        """
        Convert QASPER papers to EvaluationSamples.

        Args:
            papers: Parsed QASPER papers
            skip_unanswerable: Skip unanswerable questions
            max_samples: Limit number of samples

        Returns:
            List of EvaluationSample for evaluation
        """
        samples = []

        for paper in papers:
            for q in paper.questions:
                if skip_unanswerable and q.is_unanswerable:
                    continue

                reference = q.answers[0] if q.answers else ""

                samples.append(EvaluationSample(
                    query=q.question,
                    reference_answer=reference,
                    relevant_ids=[f"ev_{i}" for i in range(len(q.evidence))],
                    source_papers=[paper.title] if paper.title else [],
                ))

                if max_samples and len(samples) >= max_samples:
                    return samples

        logger.info(f"📊 Created {len(samples)} evaluation samples from QASPER")
        return samples

    @staticmethod
    def from_custom_dataset(
        entries: List[Dict[str, Any]]
    ) -> List[EvaluationSample]:
        """
        Create evaluation samples from a custom dataset.

        Args:
            entries: List of dicts with keys:
                - "query": str (required)
                - "answer": str (reference answer)
                - "evidence": List[str] (relevant doc IDs)
                - "source_papers": List[str]

        Returns:
            List of EvaluationSample
        """
        samples = []
        for entry in entries:
            samples.append(EvaluationSample(
                query=entry["query"],
                reference_answer=entry.get("answer", ""),
                relevant_ids=entry.get("evidence", []),
                source_papers=entry.get("source_papers", []),
            ))
        return samples

    @staticmethod
    def create_sample_dataset() -> List[Dict[str, Any]]:
        """
        Create a sample evaluation dataset for testing without QASPER.

        Returns:
            List of sample evaluation entries
        """
        return [
            {
                "query": "What F1 score did BERT achieve on SST-2?",
                "answer": "BERT achieved 93.5% F1 score on the SST-2 sentiment classification benchmark.",
                "evidence": ["results_chunk_1", "results_chunk_2"],
                "source_papers": ["BERT: Pre-training of Deep Bidirectional Transformers"]
            },
            {
                "query": "How does the transformer attention mechanism work?",
                "answer": "The transformer uses scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V",
                "evidence": ["methodology_chunk_1"],
                "source_papers": ["Attention Is All You Need"]
            },
            {
                "query": "What is the training objective of GPT-2?",
                "answer": "GPT-2 uses autoregressive language modeling, predicting the next token given previous tokens.",
                "evidence": ["training_chunk_1", "training_chunk_2"],
                "source_papers": ["Language Models are Unsupervised Multitask Learners"]
            },
            {
                "query": "Compare the pre-training approaches of BERT and GPT.",
                "answer": "BERT uses bidirectional masked language modeling while GPT uses unidirectional autoregressive modeling.",
                "evidence": ["bert_method_1", "gpt_method_1"],
                "source_papers": [
                    "BERT: Pre-training of Deep Bidirectional Transformers",
                    "Improving Language Understanding by Generative Pre-Training"
                ]
            },
            {
                "query": "What datasets were used to evaluate RoBERTa?",
                "answer": "RoBERTa was evaluated on GLUE, SQuAD, and RACE benchmarks.",
                "evidence": ["roberta_eval_1"],
                "source_papers": ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"]
            },
        ]
