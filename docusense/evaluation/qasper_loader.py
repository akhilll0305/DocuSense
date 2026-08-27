"""
QASPER Dataset Loader - Load and parse the QASPER benchmark.

PURPOSE:
--------
QASPER (Question Answering on Scientific Papers) is the standard
benchmark for evaluating QA on research papers. This loader:
1. Parses the released QASPER JSON format
2. Extracts question-answer pairs with their evidence paragraphs
3. Reconstructs each paper as a Markdown document that can be ingested
4. Converts to EvaluationSample format for our evaluator

QASPER Format (v0.3, as released):
----------------------------------
{
    "paper_id": {
        "title": "...",
        "abstract": "...",
        "full_text": [                      # a LIST, not a dict
            {"section_name": "Introduction", "paragraphs": ["...", "..."]},
            ...
        ],
        "qas": [
            {
                "question": "...",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": false,
                            "extractive_spans": [...],
                            "yes_no": null,
                            "free_form_answer": "...",
                            "evidence": ["paragraph text", ...],   # INSIDE "answer"
                            "highlighted_evidence": [...]
                        },
                        "annotation_id": "...",
                        "worker_id": "..."
                    }
                ]
            }
        ]
    }
}

Two details above were previously mis-modelled, which made the loader return
nothing usable for the real dataset: `full_text` is a list (it was read as a
dict, yielding zero sections), and `evidence` sits inside the `answer` object
(it was read as a sibling, yielding zero evidence). Both shapes are now
accepted so older hand-written fixtures keep working.

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

# Evidence entries pointing at a figure or table rather than a body paragraph.
# The reconstructed document carries body text only, so these cannot be
# grounded to a chunk and are excluded from the ground truth.
FLOAT_EVIDENCE_PREFIX = "FLOAT SELECTED"

# QASPER encodes subsection nesting in the section name, e.g.
# "Experiments ::: Automatic Evaluation Metrics".
SECTION_PATH_SEPARATOR = ":::"


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

    def paragraphs(self) -> List[str]:
        """Every body paragraph in the paper, in reading order."""
        out: List[str] = []
        for section in self.full_text_sections:
            out.extend(section.get("paragraphs", []))
        return out


class QASPERLoader:
    """
    Load and parse the QASPER benchmark dataset.

    Usage:
        loader = QASPERLoader()
        papers = loader.load("qasper-test-v0.3.json")
        markdown = loader.reconstruct_document(papers[0])
        samples = loader.to_evaluation_samples(papers)
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

        logger.info(f"Loading QASPER dataset from {file_path.name}...")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        papers = [self._parse_paper(pid, pdata) for pid, pdata in data.items()]

        total_qs = sum(len(p.questions) for p in papers)
        logger.success(f"Loaded {len(papers)} papers, {total_qs} questions")

        return papers

    def _parse_paper(self, paper_id: str, data: Dict[str, Any]) -> QASPERPaper:
        """Parse a single QASPER paper entry."""
        title = data.get("title", "") or ""
        abstract = data.get("abstract", "") or ""
        sections = self._parse_full_text(data.get("full_text"))

        questions = [
            self._parse_question(qa, paper_id, title)
            for qa in data.get("qas", [])
        ]

        return QASPERPaper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            questions=questions,
            full_text_sections=sections,
        )

    @staticmethod
    def _parse_full_text(full_text: Any) -> List[Dict[str, Any]]:
        """
        Normalize `full_text` into [{"section": name, "paragraphs": [...]}, ...].

        The released dataset uses a list of {"section_name", "paragraphs"}.
        A dict of name -> paragraphs is also accepted.
        """
        sections: List[Dict[str, Any]] = []

        def clean(paragraphs: Any) -> List[str]:
            if not isinstance(paragraphs, list):
                paragraphs = [paragraphs]
            return [
                str(p).strip()
                for p in paragraphs
                if p is not None and str(p).strip()
            ]

        if isinstance(full_text, list):
            for entry in full_text:
                if not isinstance(entry, dict):
                    continue
                sections.append({
                    "section": (entry.get("section_name") or "").strip(),
                    "paragraphs": clean(entry.get("paragraphs")),
                })
        elif isinstance(full_text, dict):
            for name, paragraphs in full_text.items():
                sections.append({
                    "section": str(name).strip(),
                    "paragraphs": clean(paragraphs),
                })

        return sections

    @staticmethod
    def _parse_question(
        qa: Dict[str, Any],
        paper_id: str,
        paper_title: str
    ) -> QASPERQuestion:
        """
        Parse one question, merging the annotations of every worker.

        QASPER questions carry several independent annotations. Evidence is
        unioned across them (a paragraph any annotator cited counts as
        evidence); the answer list keeps every distinct annotator answer.
        A question counts as unanswerable only when no annotator answered it.
        """
        answers: List[str] = []
        evidence: List[str] = []
        seen_evidence = set()
        is_unanswerable = False
        answer_type = "free_form"
        answered = False

        for ans_entry in qa.get("answers", []):
            ans_obj = ans_entry.get("answer", {}) or {}

            if ans_obj.get("unanswerable", False):
                is_unanswerable = True
                if not answered:
                    answer_type = "unanswerable"
            elif ans_obj.get("yes_no") is not None:
                answers.append("Yes" if ans_obj["yes_no"] else "No")
                answer_type = "yes_no"
                answered = True
            elif ans_obj.get("extractive_spans"):
                answers.extend(
                    s.strip() for s in ans_obj["extractive_spans"] if s and s.strip()
                )
                answer_type = "extractive"
                answered = True
            elif ans_obj.get("free_form_answer"):
                answers.append(ans_obj["free_form_answer"].strip())
                answer_type = "free_form"
                answered = True

            # Evidence lives inside the answer object in the released format;
            # an older draft placed it beside the answer. Accept both.
            raw_evidence = ans_obj.get("evidence") or ans_entry.get("evidence") or []
            for ev in raw_evidence:
                if not isinstance(ev, str):
                    continue
                ev = ev.strip()
                if ev and ev not in seen_evidence:
                    seen_evidence.add(ev)
                    evidence.append(ev)

        return QASPERQuestion(
            question=qa.get("question", ""),
            answers=answers,
            evidence=evidence,
            is_unanswerable=is_unanswerable and not answered,
            answer_type=answer_type,
            paper_id=paper_id,
            paper_title=paper_title,
        )

    @staticmethod
    def body_evidence(question: QASPERQuestion) -> List[str]:
        """
        Evidence entries that refer to body paragraphs.

        Drops "FLOAT SELECTED: ..." entries, which point at a figure or table
        rather than text and therefore have no chunk to match against.
        """
        return [
            ev for ev in question.evidence
            if not ev.startswith(FLOAT_EVIDENCE_PREFIX)
        ]

    @staticmethod
    def reconstruct_document(paper: QASPERPaper) -> str:
        """
        Rebuild a QASPER paper as Markdown, ready to ingest.

        QASPER ships parsed text rather than PDFs, so the document is
        reassembled with real headers. This keeps the ingestion path under
        measurement (chunking, section tagging, metadata extraction) identical
        to the one the product uses, and keeps evidence paragraphs verbatim so
        they can be matched back to the chunks they land in.
        """
        lines: List[str] = []

        if paper.title:
            lines += [f"# {paper.title.strip()}", ""]
        if paper.abstract:
            lines += ["## Abstract", "", paper.abstract.strip(), ""]

        for section in paper.full_text_sections:
            name = (section.get("section") or "").strip()
            if name:
                parts = [
                    p.strip()
                    for p in name.split(SECTION_PATH_SEPARATOR)
                    if p.strip()
                ]
                if parts:
                    level = min(2 + len(parts) - 1, 6)
                    lines += [f"{'#' * level} {parts[-1]}", ""]
            for paragraph in section.get("paragraphs", []):
                lines += [paragraph.strip(), ""]

        return "\n".join(lines).strip() + "\n"

    def to_evaluation_samples(
        self,
        papers: List[QASPERPaper],
        skip_unanswerable: bool = True,
        max_samples: Optional[int] = None
    ) -> List[EvaluationSample]:
        """
        Convert QASPER papers to EvaluationSamples.

        `relevant_ids` is deliberately left empty here: relevance is defined
        over the chunk ids produced by ingestion, which do not exist until the
        paper has been ingested. The evidence text is carried on the sample so
        `QASPERHarness` can resolve it to real chunk ids. Filling relevant_ids
        with synthetic placeholders (as an earlier version did) scores every
        retrieval metric at exactly zero, because no placeholder can ever match
        a retrieved chunk id.

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
                    relevant_ids=[],
                    evidence_texts=self.body_evidence(q),
                    source_papers=[paper.title] if paper.title else [],
                    paper_id=paper.paper_id,
                ))

                if max_samples and len(samples) >= max_samples:
                    return samples

        logger.info(f"Created {len(samples)} evaluation samples from QASPER")
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

        Note: these carry hand-written relevant ids and no ingested corpus, so
        they exercise the metric plumbing only. Real numbers come from
        `QASPERHarness`.

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
